# Port Plan — `experimental/paged_cache`

Port plan for `ttnn/cpp/ttnn/operations/experimental/paged_cache`, ported from the
`ProgramDescriptorFactoryConcept` (`create_descriptor`) API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

---

## Scope — two passes

The audit brief scopes the op as **three `DeviceOperation`s, eight factories, four program bodies**.
Those eight split into three groups, each with a different disposition. Pass 1 ported the two
non-fused single-device factories; **pass 2 (this one) ports the two fused single-device factories**,
which is every factory this recipe's procedure covers.

| Factory | Selected when | Status |
|---|---|---|
| `PagedUpdateCacheProgramFactory` | `mesh_coords == nullopt` | **PORTED** (pass 1) |
| `PagedFillCacheProgramFactory` | `mesh_coords == nullopt` | **PORTED** (pass 1) |
| `PagedTiledFusedUpdateCacheProgramFactory` | `mesh_coords == nullopt` | **PORTED** (pass 2) |
| `PagedRowMajorFusedUpdateCacheProgramFactory` | `mesh_coords == nullopt` | **PORTED** (pass 2) |
| `PagedUpdateCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **OUT OF PROCEDURE** — target concept `MeshWorkloadSpecFactoryConcept` |
| `PagedFillCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **OUT OF PROCEDURE** — same |
| `PagedTiledFusedUpdateCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **OUT OF PROCEDURE** — same |
| `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **OUT OF PROCEDURE** — same |

**Why the four mesh factories are not in pass 2, even though their framework blocker cleared.**
Pass 1 recorded them as blocked on a framework capability that did not exist: a per-mesh-coordinate
`ProgramSpec` / `ProgramRunArgs`. That capability **has since merged** — `9fb0ed54794`
*"Let a spec factory build a different program per mesh coordinate (#54988)"* ships
`MeshWorkloadSpecFactoryConcept` and `MeshWorkloadArtifacts::PerCoordProgram`
(`ttnn/api/ttnn/metal_v2_artifacts.hpp:38-52`, `ttnn/api/ttnn/operation_concepts.hpp:118-132`), and it
is exactly the vehicle pass 1 asked for. So these four are no longer *blocked*.

They are, however, outside **this procedure's** coverage boundary, which is a different thing. The
port recipe states its scope as `ProgramSpecFactoryConcept` or `CustomProgramSpecFactoryConcept` —
"the two single-program Metal 2.0 concepts, one spec stamped across the mesh" — and rules that a brief
naming any other target concept is out of scope, naming this exact case: *"The case to expect is a
mesh-workload concept for a genuine multi-program op: the audit can clear one the day TTNN support
lands, but no port procedure exists for it until someone writes it. Stopping is the correct outcome;
improvising a multi-program port out of this recipe is not."* The audit-side doc says the same from
the other end: if the target is a mesh-workload concept, *"the port procedure does not cover it yet,
so the porter will stop at its coverage boundary."*

So pass 2 stops there rather than improvising, and the four are carried in
`METAL2_PORT_REPORT.md` → *Handoff points* #1 as **ready to port, awaiting a procedure**, with the
`MeshWorkloadSpecFactoryConcept` design work already done in that entry.

**Kernel forks.** Because each `*MeshWorkloadFactory` binds the same kernel sources as its
single-device sibling and cannot convert with it, every ported factory takes the **intra-op fork**
rung of *Caution: Porting a shared kernel*. Pass 1 created five `_metal2` forks; pass 2 creates
**six** more (the fused readers, writers and compute kernels), for eleven in total — one per kernel
source in the op. The legacy originals stay untouched apart from the mandated pointer comment and
keep serving the four mesh factories.

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

### Variant: `PagedTiledFusedUpdateCacheProgramFactory` (`paged_tiled_fused_update_cache_program_factory.cpp`)

Line references are to the **pre-pass-2** file (`git merge-base origin/main HEAD`).

#### Kernels

All three kernels are created over `all_cores_bb` — the *bounding box* of
`input1_cores ∪ input2_cores`, which can be strictly larger than that union.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp` | `all_cores_bb` | 0 `src1_cb_index`, 1 `src2_cb_index`, 2 `cache_cb_index`, 3 `use_index_tensor`, 4 `index_is_dram`, 5 `cb_index_id`, 6 `cache_batch_num_tiles`, 7 `Wt`, 8 `log2_page_size`, 9 `index_stick_size`, 10 `is_paged_cache`, 11 `num_heads`, 12 `block_size`, 13 `block_size_t`, 14 `max_blocks_per_seq`, 15 `log2_page_table_stick_size`, 16 `page_table_stick_size`, 17 `page_table_is_dram`, 18 `cb_pagetable_id`, 19 `St`, 20 `in0_sequential_mode_semaphore_id`, 21 `B`, then `TensorAccessorArgs` ×3 (cache1, index, page_table) | none | working cores (8): `[0]`=`has_work`, `[1]`=`is_input1`, `[2]`=`Buffer*` cache1 **\|** cache2, `[3]`=`cache_start_id`, `[4]`=`Buffer*` index \| `0`, `[5]`=`i`, `[6]`=`Buffer*` page_table \| `0`, `[7]`=`wait_to_start`. `unused_cores` (1): `[0]`=`!has_work` | none | none | O2 | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp` | `all_cores_bb` | 0 `output_cb_index`, 1 `intermed0_cb_index`, 2 `intermed1_cb_index`, 3 `intermed2_cb_index`, 4 `use_index_tensor`, 5 `cb_index_id`, 6 `cache_batch_num_tiles`, 7 `Wt`, 8 `Wbytes`, 9 `is_paged_cache`, 10 `num_heads`, 11 `block_size`, 12 `block_size_t`, 13 `max_blocks_per_seq`, 14 `cb_pagetable_id`, 15 `St`, 16 `in0_sequential_mode_semaphore_id`, 17 `B`, 18 `page_table_stick_size`, 19 `page_table_is_dram`, then `TensorAccessorArgs` (cache1) | none | working cores (8): `[0]`=`has_work`, `[1]`=`Buffer*` cache1 **\|** cache2, `[2]`=`cache_start_id`, `[3]`=`tile_update_offset_B`, `[4]`=`i`, `[5]`=`send_signal`, `[6]`=`send_core_x`, `[7]`=`send_core_y`. `unused_cores` (1): `[0]`=`!has_work` | none | none | O2 | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/paged_fused_update_cache.cpp` | `all_cores_bb` | 0 `src1_cb_index`, 1 `src2_cb_index`, 2 `cache_cb_index`, 3 `intermed0_cb_index`, 4 `intermed1_cb_index`, 5 `intermed2_cb_index`, 6 `output_cb_index`, 7 `Wt`, 8 `num_heads` | none | working cores (2): `[0]`=`has_work`, `[1]`=`is_input1`. `unused_cores` (1): `[0]`=`!has_work` | none | none | **O3** (resolved — `ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en}` |

`grep -n opt_level` over the file returns **nothing**, so all three resolve to their legacy
per-kernel-type default: `O2` on the two DM kernels, **`O3`** on compute.

The **per-core runtime-arg count varies inside one `KernelDescriptor`** (8/8/2 on working cores,
**1** on `unused_cores`). `unused_cores` is non-empty only when `input1_cores ∪ input2_cores` is not
itself a rectangle.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | `.buffer` |
|---|---|---|---|---|---|---|
| `c_0` `cache` | `num_cache_tiles * cache_single_tile_size` (`num_cache_tiles = 2*Wt`) | `all_cores_bb` | `cache_cb_data_format` | `cache_single_tile_size` | — | — |
| `c_1` `src1` | `num_input_tiles * input_single_tile_size` | **`input1_cores`** | `input_cb_data_format` | `input_single_tile_size` | — | `in1_buffer` (**borrowed**) |
| `c_2` `src2` | `num_input_tiles * input_single_tile_size` | **`input2_cores`** | `input_cb_data_format` | `input_single_tile_size` | — | `in2_buffer` (**borrowed**) |
| `c_3` `index` | `index_stick_size` | `all_cores_bb` | `index_data_format` | `index_stick_size` | — | `index_buffer_ptr` — non-null **only when the index tensor is sharded** (`:117`) |
| `c_4` `page_table` | `num_pages_page_table * page_table_stick_size` | `all_cores_bb` | `page_table_data_format` | `page_table_stick_size` | — | `page_table_buffer_ptr` — non-null **only when the page table is sharded** (`:136`) |
| `c_24` `intermed0` + `c_25` `intermed1` | `num_interm_tiles * interm_single_tile_size` | `all_cores_bb` | `interm_cb_data_format` | `interm_single_tile_size` | — | — |
| `c_26` `intermed2` | `num_interm_tiles * interm_single_tile_size` | `all_cores_bb` | `interm_cb_data_format` | `interm_single_tile_size` | — | — |
| `c_16` `output` | `num_output_tiles * cache_single_tile_size` (`num_output_tiles = B*Wt`) | `all_cores_bb` | `cache_cb_data_format` | `cache_single_tile_size` | — | — |

`c_24`/`c_25` are **one** `CBDescriptor` with **two** `CBFormatDescriptor`s (`:223-238`) — two buffer
indices aliasing one L1 allocation. That aliasing is the algorithm, not an optimisation (see
*Applied Patterns*). `c_3` allocated only `if (use_index_tensor)` (`:267-278`); `c_4` only
`if (is_paged_cache)` (`:280-291`). No `GlobalCircularBuffer`.

Endpoint census, re-derived from the kernels (agrees with the audit — all 1P+1C, no self-loop, no
multi-binding):

| CB | verdict | producer | consumer |
|---|---|---|---|
| `c_0` `cache` | **1P+1C** | reader `reserve_back`/`get_write_ptr`/`push_back` @`reader:158-169` | compute `untilize<Wt, cache_cb, …>` @`compute:59` |
| `c_1` `src1` | **1P+1C** | reader `reserve_back`/`push_back` @`reader:73-74` | compute `untilize<Wt, in1_cb, …>` @`compute:48-54` |
| `c_2` `src2` | **1P+1C** | reader, same two lines (runtime-selected index) | compute `untilize<Wt, in2_cb, …>` @`compute:40-46` |
| `c_3` `index` | **1P+1C** | reader `reserve_back`/`push_back` @`reader:88-95` | writer `wait_front`/`pop_front` @`writer:77,122` |
| `c_4` `page_table` | **1P+1C** | reader @`reader:105-121` | writer @`writer:88,112` |
| `c_24` `intermed0` | **1P+1C** | compute `untilize` output @`compute:59` | writer `wait_front`/`pop_front` @`writer:134,146` |
| `c_25` `intermed1` | **1P+1C** | writer `reserve_back`/`push_back` @`writer:135,145` | compute `tilize` input @`compute:62` |
| `c_26` `intermed2` | **1P+1C** | compute `untilize` output @`compute:40-54` | writer `wait_front`/`pop_front` @`writer:125,172` |
| `c_16` `output` | **1P+1C** | compute `tilize` output @`compute:62` | writer (kernel calls it `cache_cb_id`) `wait_front`/`pop_front` @`writer:149,160` |

**Counter-intuitive name, confirmed:** the writer's CTA-0 `cache_cb_id` is the **output** CB `c_16`,
not the cache CB `c_0` — which the writer never touches. Resolved through the factory's arg list
(`:328`), not the kernel's local name.

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`in0_sequential_mode_semaphore_id`) | `WORKER` | `all_cores_bb` | 0 |

`share_cache` chain: writer *i* signals reader *i+1* on **both** core lists —
`Semaphore<>(receiver_sem_id).up(noc, send_core_x, send_core_y, 1)` @`writer:176`, awaited at
`reader:152-154`. `send_core{1,2}_{x,y}` are **physical** coordinates baked host-side via
`worker_core_from_logical_core` (`:422,427`). The `noc.async_atomic_barrier()` at `writer:182`
documents a real Watcher NOC-idle race — **preserved verbatim**.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:322` (reader CTAs) | `cache_tensor1` — and `cache_tensor2` through the **same** slot on `cores2` | reader RTA[2] (`:438` cores1 / `:483` cores2) |
| `:323-324` (reader CTAs) | `update_idxs_tensor` (optional) | reader RTA[4] (`:441` / `:486`) |
| `:325` (reader CTAs) | `page_table` (optional) | reader RTA[6] (`:447` / `:492`) |
| `:350` (writer CTAs) | `cache_tensor1` / `cache_tensor2`, same overloading | writer RTA[1] (`:456-467` / `:501-512`) |

All four are 2-arg constructions, all **Case 1** — every tensor base feeds a `TensorAccessor`;
nothing does hand-rolled NoC arithmetic on a tensor base, so **no `get_bank_base_address` bridge**.
The raw pointer arithmetic that does exist (`page_table_cb_wr_ptr += my_batch_idx *
page_table_stick_size` @`reader:118`, `writer:91`) walks an **L1 CB** pointer, not a tensor base.

Note that `TensorAccessorArgs` is appended for the index and page-table tensors **whenever the
tensor is present**, regardless of sharding, while the *accessor* is only constructed — and only used
— on the DRAM path (`reader:87` unconditional but read-gated at `:90`; `reader:109` inside
`if constexpr (page_table_is_dram)`).

#### Work split

- **No `split_work_to_cores`.** The core lists come from the two input tensors' shard grids:
  `cores1 = corerange_to_cores(input1_cores, …, row_major)`, `cores2` likewise (`:396-397`), and the
  device op validates them **disjoint** and **equal in count**
  (`paged_fused_update_cache_device_operation.cpp:351-357`).
- Index *i* handles input1 on `cores1[i]` (writing `cache_tensor1`) and input2 on `cores2[i]`
  (writing `cache_tensor2`); both share the same `cache_start_id` / `tile_update_offset_B`.
- `all_cores = input1_cores.merge(input2_cores)`; `all_cores_bb = all_cores.bounding_box()`;
  `unused_cores = all_cores_bb.subtract(all_cores)`.
- No per-group CTA anywhere, so no multiplicity to preserve.

---

### Variant: `PagedRowMajorFusedUpdateCacheProgramFactory` (`paged_row_major_fused_update_cache_program_factory.cpp`)

Structurally the tiled variant with **one fewer intermediate buffer**. `diff -u` against the tiled
factory is 5 substantive hunks; everything else is `const`-qualification and identifier renames.

#### Kernels

Same three roles over `all_cores_bb`, same `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` /
`ComputeConfigDescriptor{.fp32_dest_acc_en = …}`, same resolved `opt_level` (O2/O2/**O3**,
`grep -n opt_level` empty). Differences from tiled:

| | difference |
|---|---|
| reader | identical CTA list and RTA list; kernel differs only in `reserve_back(1)`/`push_back(1)` instead of `(Wt)` on the input DFB (`reader_paged_row_major_…:73-74`) |
| writer | CTA slot 3 becomes **`src1_cb_index`** and a new slot 4 **`src2_cb_index`** (there is no `intermed2`), shifting every later slot by one; RTA gains a **9th** arg `is_input1` (`:464`, `:510`) |
| compute | source `device/kernels/compute/paged_row_major_fused_update_cache.cpp`; CTA list drops `intermed2_cb_index` (8 args, not 9) |

#### CBs

Same as tiled minus `c_26`, and the intermediates renumber:

| index | total_size | core_ranges | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| `c_0` `cache` | `num_cache_tiles * cache_single_tile_size` | `all_cores_bb` | `cache_cb_data_format` | `cache_single_tile_size` | — |
| `c_1` `src1` | `num_input_tiles * input_single_tile_size` | **`input1_cores`** | `input_cb_data_format` | `input_single_tile_size` | `in1_buffer` (**borrowed**) |
| `c_2` `src2` | `num_input_tiles * input_single_tile_size` | **`input2_cores`** | `input_cb_data_format` | `input_single_tile_size` | `in2_buffer` (**borrowed**) |
| `c_3` `index` | `index_stick_size` | `all_cores_bb` | `index_data_format` | `index_stick_size` | sharded-only |
| `c_4` `page_table` | `num_pages_page_table * page_table_stick_size` | `all_cores_bb` | `page_table_data_format` | `page_table_stick_size` | sharded-only |
| `c_5` `intermed0` + `c_6` `intermed1` | `num_interm_tiles * interm_single_tile_size` | `all_cores_bb` | `interm_cb_data_format` | `interm_single_tile_size` | — |
| `c_7` `output` | `num_output_tiles * cache_single_tile_size` | `all_cores_bb` | `cache_cb_data_format` | `cache_single_tile_size` | — |

`c_5`/`c_6` are the aliased pair (one `CBDescriptor`, two format descriptors, `:228-243`).

Endpoint census — **the input CBs move consumer**, which is the one census difference that matters:

| CB | verdict | producer | consumer |
|---|---|---|---|
| `c_0` `cache` | **1P+1C** | reader @`reader:158-169` | compute `untilize<Wt, cache_cb, …>` @`compute:39` |
| `c_1` `src1` | **1P+1C** | reader @`reader:73-74` | **writer** @`writer:130,174` (`untilized_input_cb_id`) |
| `c_2` `src2` | **1P+1C** | reader, same lines | **writer**, same lines (runtime-selected index) |
| `c_3` `index` | **1P+1C** | reader @`reader:92-99` | writer `wait_front` @`writer:84` (no `pop_front` — fill-once/read-once, intentional) |
| `c_4` `page_table` | **1P+1C** | reader @`reader:109-125` | writer `wait_front` @`writer:95` (no `pop_front`) |
| `c_5` `intermed0` | **1P+1C** | compute @`compute:39` | writer @`writer:136,148` |
| `c_6` `intermed1` | **1P+1C** | writer @`writer:137,147` | compute `tilize` @`compute:42` |
| `c_7` `output` | **1P+1C** | compute @`compute:42` | writer @`writer:151,162` |

Because the row-major input needs no untilize step, **compute never touches `src1`/`src2`** here —
its `in1_cb` / `in2_cb` / `is_input1` are the dead `[[maybe_unused]]` args the audit catalogued. The
untilized input the writer consumes *is* the input CB.

The missing `pop_front` on `c_3` / `c_4` is **not** an unbalanced FIFO to fix: the buffers are
filled once and read once per dispatch and per-execution DFB state is reinitialised. It is a
behaviour difference from the tiled writer (which does pop both) that the port preserves.

#### Semaphores / Tensor accessors / Work split

Identical to the tiled variant apart from line numbers (semaphore `:256`; reader accessor CTAs
`:319-321`; writer accessor CTA `:347`; cores `:393-394`).

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
| `dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp` | `PagedTiledFusedUpdateCacheProgramFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory` | no | **2 — create the fork** (pass 2) |
| `dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp` | same pair | no | **2 — create the fork** (pass 2) |
| `compute/paged_fused_update_cache.cpp` | same pair | no | **2 — create the fork** (pass 2) |
| `dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | `PagedRowMajorFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` | no | **2 — create the fork** (pass 2) |
| `dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | same pair | no | **2 — create the fork** (pass 2) |
| `compute/paged_row_major_fused_update_cache.cpp` | same pair | no | **2 — create the fork** (pass 2) |

Eleven forks in total across the two passes — one per kernel source in the op. Remaining consumers
of every original: the four `*MeshWorkloadFactory` factories (out of procedure — see
[Deferred / Flagged](#deferred--flagged)). Recorded in `METAL2_PORT_REPORT.md` →
*Open items for downstream*.

`ls` of each original's directory confirmed no `_metal2` fork existed beside any of the six pass-2
kernels beforehand (a locational check, not a tree-wide grep). No build-system change was needed:
the op's kernels are installed by a `file(GLOB_RECURSE …)` that already covers both directories.

### Flags

- **No unreferenced kernel files.** All 11 sources under `device/kernels/` are bound by a factory.
- The dead CTAs the audit catalogued (`log_base_2_of_page_size`, `log2_page_table_stick_size`,
  `max_blocks_per_seq`) are **carried through as named args**, not removed — dropping one would
  change the arg schema, which is a functional change the port is not entitled to make.
- `paged_fill_cache_program_factory.cpp:116` carries a `TT_FATAL` inside the factory body; it is
  preserved verbatim in the ported body (TT_FATAL census below).

Pass 2 adds:

- **No unreferenced kernel files still.** All 17 sources under `device/kernels/` (11 legacy + 6 new
  forks; 11 + 5 after pass 1) are bound by a factory.
- **The row-major fused factory's dead `[[maybe_unused]]` compute args are asymmetric with the tiled
  one's.** The tiled compute kernel genuinely uses `in1_cb` / `in2_cb` / `is_input1`; the row-major
  one reads all three and uses none, because a row-major input needs no untilize step. Only the
  scalar (`is_input1`) is carried forward — see Dropped Plumbing for why the two buffer-index args
  could not be.
- **The two fused variants' `page_table_stick_size` is computed differently from `update_cache`'s.**
  Both fused factories use `page_table.value().buffer()->aligned_page_size()` (tiled `:141`, RM `:141`);
  `update_cache` uses `page_table_tensor.padded_shape()[-1] * page_table_tensor.element_size()`
  (`:171`). Not the port's to reconcile — noted so a reader does not "fix" one to match the other.
- **`num_pages_page_table` is `B` when the page table is sharded and `1` when it is not** (tiled
  `:137`), which is why the page-table DFB's `num_entries` is not simply 1 as it is in `update_cache`.
- **The row-major fused writer does not `pop_front` its index or page-table DFB, while the tiled one
  does.** Preserved as-is on both. Not an unbalanced FIFO — fill-once/read-once per dispatch, and
  per-execution DFB state is reinitialised.

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

### Variant: `PagedTiledFusedUpdateCacheProgramFactory`

Spec resource names are prefixed `TF_` in the factory's anonymous namespace (the `.cpp` files are
unity-built into one translation unit, so the prefixes keep the names from colliding with the
`UC_` / `FC_` / `RMF_` sets).

- **KernelSpecs** (3, 1:1 with legacy — no work-split multiplicity):
  - `TF_READER_KERNEL` — `.../reader_paged_fused_update_cache_interleaved_start_id_metal2.cpp`,
    `hw_config = create_reader_datamovement_config(arch)` (legacy `ReaderConfigDescriptor{}` resolves
    to `RISCV_1 / NOC_0 / DM_DEDICATED_NOC`), `compiler_options.opt_level` left at the default `O2`.
  - `TF_WRITER_KERNEL` — `.../writer_paged_fused_update_cache_interleaved_start_id_metal2.cpp`,
    `hw_config = create_writer_datamovement_config(arch)`, `opt_level` default `O2`.
  - `TF_COMPUTE_KERNEL` — `.../compute/paged_fused_update_cache_metal2.cpp`,
    `hw_config = ComputeGen1Config{.enable_32_bit_dest = fp32_dest_acc_en, .unpack_modes = …}`,
    **`compiler_options.opt_level = O3` set explicitly** (legacy `ComputeConfigDescriptor` resolves
    to O3; Metal 2.0 `CompilerOptions` defaults to O2).
- **DataflowBufferSpecs** (9, one per legacy `buffer_index`):

  | name | legacy | entry_size | num_entries | data_format | notes |
  |---|---|---|---|---|---|
  | `TF_CACHE_DFB` | `c_0` | `cache_single_tile_size` | `num_cache_tiles` | `cache_data_format` | |
  | `TF_SRC1_DFB` | `c_1` | `input_single_tile_size` | `num_input_tiles` | `input_data_format` | `borrowed_from = TF_INPUT1_TENSOR` |
  | `TF_SRC2_DFB` | `c_2` | `input_single_tile_size` | `num_input_tiles` | `input_data_format` | `borrowed_from = TF_INPUT2_TENSOR` |
  | `TF_INDEX_DFB` | `c_3` | `index_stick_size` | 1 | `index_data_format` | conditional on `use_index_tensor`; `borrowed_from = TF_INDEX_TENSOR` **only when the index tensor is sharded** |
  | `TF_PAGE_TABLE_DFB` | `c_4` | `page_table_stick_size` | `num_pages_page_table` | `page_table_data_format` | conditional on `is_paged_cache`; `borrowed_from = TF_PAGE_TABLE_TENSOR` **only when the page table is sharded** |
  | `TF_UNTILIZED_CACHE_DFB` | `c_24` | `interm_single_tile_size` | `num_interm_tiles` | `interm_data_format` | `alias_with = {TF_UNTILIZED_CACHE2_DFB}` |
  | `TF_UNTILIZED_CACHE2_DFB` | `c_25` | `interm_single_tile_size` | `num_interm_tiles` | `interm_data_format` | `alias_with = {TF_UNTILIZED_CACHE_DFB}` |
  | `TF_UNTILIZED_INPUT_DFB` | `c_26` | `interm_single_tile_size` | `num_interm_tiles` | `interm_data_format` | |
  | `TF_OUTPUT_DFB` | `c_16` | `cache_single_tile_size` | `num_output_tiles` | `cache_data_format` | |

  `tile_format_metadata` left unset on all nine — no legacy `CBFormatDescriptor` set `.tile`.

  The two **conditional** `borrowed_from`s mirror the legacy `.buffer = <ptr>` exactly, where the
  pointer is `nullptr` unless the tensor is sharded (`:117`, `:136`). On the DRAM path the DFB is a
  normal L1 allocation the reader NoC-reads into; on the L1-sharded path it is a view over the
  resident tensor and the reader's read compiles out. Both paths must survive, and this is the field
  that carries the distinction.
- **SemaphoreSpecs** (1): `TF_SEQUENTIAL_MODE_SEM`, `target_nodes = all_cores_bb`.
- **TensorParameters** (6, one per distinct originating tensor):
  `TF_CACHE1_TENSOR`, `TF_CACHE2_TENSOR` (each bound by READER **and** WRITER — two `TensorBinding`s
  per parameter), `TF_INPUT1_TENSOR` / `TF_INPUT2_TENSOR` (no kernel binding — they exist solely as
  the `borrowed_from` targets of `TF_SRC1_DFB` / `TF_SRC2_DFB`), `TF_INDEX_TENSOR` (conditional,
  READER only), `TF_PAGE_TABLE_TENSOR` (conditional, READER only, and *only on the DRAM path* —
  see *Dropped Plumbing*).
- **WorkUnitSpecs** (1): `{TF_READER_KERNEL, TF_WRITER_KERNEL, TF_COMPUTE_KERNEL}` over
  `all_cores_bb`. One work unit, matching legacy's single `core_ranges = all_cores_bb` on all three
  `KernelDescriptor`s — so every DFB's derived node set is `all_cores_bb`, which is what makes the
  alias group legal (all members must target the same node set).
- **Op-owned tensors**: none.

DFB endpoint roles — all nine are 1P+1C on every node; no self-loop, no
`allow_instance_multi_binding`:

| DFB | PRODUCER | CONSUMER |
|---|---|---|
| `TF_CACHE_DFB` | READER | COMPUTE |
| `TF_SRC1_DFB` | READER | COMPUTE |
| `TF_SRC2_DFB` | READER | COMPUTE |
| `TF_INDEX_DFB` | READER | WRITER |
| `TF_PAGE_TABLE_DFB` | READER | WRITER |
| `TF_UNTILIZED_CACHE_DFB` | COMPUTE | WRITER |
| `TF_UNTILIZED_CACHE2_DFB` | WRITER | COMPUTE |
| `TF_UNTILIZED_INPUT_DFB` | COMPUTE | WRITER |
| `TF_OUTPUT_DFB` | COMPUTE | WRITER |

**`unpack_modes`.** `enable_32_bit_dest = fp32_dest_acc_en`. When set, every DFB the compute kernel
**consumes** whose `data_format_metadata` is `Float32` needs an explicit entry, and legacy's
`unpack_to_dest_mode` was empty (all `Default`), so each entry is `UnpackMode::UnpackToSrc`. The
consumed DFBs are `TF_SRC1_DFB`, `TF_SRC2_DFB`, `TF_CACHE_DFB` and `TF_UNTILIZED_CACHE2_DFB`; the
factory computes the set from the resolved data formats rather than hardcoding it. Note
`interm_data_format` is `Float32` **exactly when** `fp32_dest_acc_en`, so the
`TF_UNTILIZED_CACHE2_DFB` entry is always present in that mode even though no *tensor* is Float32.

**The runtime-selected input DFB (audit Question #1, DFB half).** Both `TF_SRC1_DFB` and
`TF_SRC2_DFB` are bound **unconditionally** to the READER and the COMPUTE `KernelSpec`s, and the
kernel selects which token to build from at runtime off the existing `is_input1` arg. Rationale and
the four legality checks are in `METAL2_PORT_REPORT.md` → *Open items* #1; the load-bearing facts
are that a borrowed DFB short-circuits allocation (so widening its node set costs no L1) and that
both borrowed-DFB checks are whole-buffer rather than per-node.

**The one honest delta from legacy.** Legacy configured `c_1` only over `input1_cores` and `c_2` only
over `input2_cores` (validated disjoint). Metal 2.0 derives placement from bindings, so each is now
configured over all of `all_cores_bb`, and on the half where legacy left it unconfigured it carries
the one program-wide borrowed base address — which there points into the *other* input tensor's
region. **Inert on Gen1**: the `is_input1` guard means those nodes never touch it and there is no
allocation to collide with. Recorded as Quasar-uplift debt, with a comment at the binding site.

### Variant: `PagedRowMajorFusedUpdateCacheProgramFactory`

Prefix `RMF_`. Same shape as the tiled variant with these differences:

- **KernelSpecs** (3): sources are the row-major `_metal2` forks; `RMF_COMPUTE_KERNEL` is
  `.../compute/paged_row_major_fused_update_cache_metal2.cpp`. Same `hw_config` derivation, same
  explicit `O3` on compute.
- **DataflowBufferSpecs** (8) — the tiled set minus `TF_UNTILIZED_INPUT_DFB`, with the intermediates
  renumbered (`c_5`/`c_6` aliased pair, `c_7` output). Sizes and formats are computed identically.
- **TensorParameters** (6): same six.
- **SemaphoreSpecs** (1), **WorkUnitSpecs** (1) over `all_cores_bb`: same.

DFB endpoint roles — the input DFBs' **consumer is the WRITER, not COMPUTE** (a row-major input needs
no untilize step, so the "untilized input" the writer consumes *is* the input buffer):

| DFB | PRODUCER | CONSUMER |
|---|---|---|
| `RMF_CACHE_DFB` | READER | COMPUTE |
| `RMF_SRC1_DFB` | READER | **WRITER** |
| `RMF_SRC2_DFB` | READER | **WRITER** |
| `RMF_INDEX_DFB` | READER | WRITER |
| `RMF_PAGE_TABLE_DFB` | READER | WRITER |
| `RMF_UNTILIZED_CACHE_DFB` | COMPUTE | WRITER |
| `RMF_UNTILIZED_CACHE2_DFB` | WRITER | COMPUTE |
| `RMF_OUTPUT_DFB` | COMPUTE | WRITER |

So on this variant the runtime-selected-input binding pair goes on the READER and the **WRITER**;
COMPUTE binds neither, because it never touches them.

**`unpack_modes`** consumed set is therefore smaller: `RMF_CACHE_DFB` and
`RMF_UNTILIZED_CACHE2_DFB` only.

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

### `PagedTiledFusedUpdateCacheProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA[2] (`:438` cores1 / `:483` cores2), writer RTA[1] (`:456-467` / `:501-512`); patched `paged_fused_update_cache_device_operation.cpp:108,118` | **one** `Buffer*` slot carrying `dst1_buffer` on `cores1` and `dst2_buffer` on `cores2` | **two** `TensorBinding`s per kernel — `{TF_CACHE1_TENSOR, "cache1"}` and `{TF_CACHE2_TENSOR, "cache2"}` — both bound on both DM kernels. The kernel branches on the existing `is_input1` RTA and instantiates the shared loop body with the matching typed accessor (audit Question #1, tensor half; `TensorBindingToken` carries its identity in the *type*, so a ternary cannot work) |
| reader RTA[4] (`:441` / `:486`); patched `:111` | `Buffer*` update_idxs, or literal `0` | conditional `TensorBinding{TF_INDEX_TENSOR, "index"}`; the literal-`0` alternative has **no** scalar counterpart (the kernel never reads it — access is behind the gate) |
| reader RTA[6] (`:447` / `:492`); patched `:114` | `Buffer*` page_table, or literal `0` | conditional `TensorBinding{TF_PAGE_TABLE_TENSOR, "page_table"}`, declared **only on the DRAM path** — on the L1-sharded path legacy passed the address but no kernel code reads it (`reader:109` is inside `if constexpr (page_table_is_dram)`), so there is no accessor to bind. Same — no scalar counterpart |
| CB `c_1` `.buffer = in1_buffer` (`:211`), CB `c_2` `.buffer = in2_buffer` (`:221`); re-pointed at `paged_fused_update_cache_device_operation.cpp:73-74` via `UpdateDynamicCircularBufferAddress` | globally-allocated CBs backed by the two input shards | `DataflowBufferSpec::borrowed_from = TF_INPUT1_TENSOR` / `TF_INPUT2_TENSOR`, refreshed by those parameters' `TensorArgument`s |
| CB `c_3` `.buffer = index_buffer_ptr` (`:276`), CB `c_4` `.buffer = page_table_buffer_ptr` (`:289`); re-pointed at `:77,80` | CB backed by the tensor **only when sharded** (`nullptr` otherwise) | `borrowed_from` set on the same condition; the `nullptr` case is a plain L1 allocation with no `borrowed_from` |
| reader CTA 0/1/2/5/18 (`:298-300,304,318`) | `src1_cb_index`, `src2_cb_index`, `cache_cb_index`, `cb_index_id`, `cb_pagetable_id` | `DFBBinding`s → `dfb::src1`, `dfb::src2`, `dfb::cache`, `dfb::index`, `dfb::page_table` |
| writer CTA 0/1/2/3/5/14 (`:328-331,334,344`) | `output_cb_index`, `intermed0/1/2_cb_index`, `cb_index_id`, `cb_pagetable_id` | `DFBBinding`s → `dfb::cache` (the **output** DFB under the kernel's own name), `dfb::untilized_cache`, `dfb::untilized_cache2`, `dfb::untilized_input`, `dfb::index`, `dfb::page_table` |
| compute CTA 0..6 (`:353-359`) | seven CB indices | `DFBBinding`s → `dfb::src1`, `dfb::src2`, `dfb::cache`, `dfb::untilized_cache`, `dfb::untilized_cache2`, `dfb::untilized_in`, `dfb::out` |
| reader CTA 20 (`:320`), writer CTA 16 (`:346`) | `in0_sequential_mode_semaphore_id` | `SemaphoreBinding{TF_SEQUENTIAL_MODE_SEM, "receiver"}` → `sem::receiver` |
| reader CTAs from `:322-325`, writer CTA from `:350` | `TensorAccessorArgs(buffer).append_to(cta)` ×4, with the kernel-side `TensorAccessorArgs<22>()` / `next_compile_time_args_offset()` chain (`reader:61-63`) and `TensorAccessorArgs<20>()` (`writer:56`) | the binding mechanism end-to-end; kernel writes `TensorAccessor(tensor::cache1)` etc. |
| reader CTA 3 (`:302`), writer CTA 4 (`:333`) | `use_index_tensor` gating `if constexpr` blocks that name a conditionally-bound DFB / tensor | `compiler_options.defines["USE_INDEX_TENSOR"]` + kernel-side `#ifdef` |
| reader CTA 10 (`:310`), writer CTA 9 (`:339`) | `is_paged_cache`, same shape | `compiler_options.defines["IS_PAGED_CACHE"]` + kernel-side `#ifdef` |
| reader CTA 17 (`:317`) | `page_table_is_dram` — **also** gates whether `tensor::page_table` is bound (see above) | `compiler_options.defines["PAGE_TABLE_IS_DRAM"]` + kernel-side `#ifdef`, *and* retained as a named CTA on the **writer**, where it gates only pointer arithmetic and no binding |
| every remaining positional CTA | positional `compile_time_args` vector | named `compile_time_args = {{name, value}, …}` |
| the whole `has_work` short-arg convention on `unused_cores` (`:524-530`) | **one** RTA on those nodes vs. 8/8/2 on working nodes | a `runtime_arg_schema` is one schema for the whole `KernelSpec` and `SetProgramRunArgs` requires **every** declared name on **every** node the kernel runs on (`program_run_args.cpp:296-324`), so `unused_cores` nodes get the full named set with `has_work = 0` and don't-care `0`s. The kernels early-return on `has_work` and never read the rest, so this is zero-functional-change; narrowing the `KernelSpec`'s node set instead would change which nodes get kernels and DFBs, which legacy fixed at `all_cores_bb` |

No page-size third-argument CTA/RTA (all accessors are 2-arg) and no semaphore-ID RTA.

`index_is_dram` (reader CTA 4) stays a **named CTA**, not a define: it gates only the NoC read, and
the `tensor::index` accessor it guards is constructed unconditionally under `USE_INDEX_TENSOR`
exactly as legacy constructed it (`reader:87`), so no name it references can be absent.

### `PagedRowMajorFusedUpdateCacheProgramFactory`

Identical table with the row-major line numbers and these three differences:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| writer CTA 3/4 (`:324-325`) | `src1_cb_index`, `src2_cb_index` (there is no `intermed2`) | `DFBBinding`s → `dfb::src1`, `dfb::src2` on the **writer**, which is this variant's consumer of the input buffers |
| writer RTA[8] (`:464` / `:510`) | `is_input1` | stays a **named RTA** `is_input1`; it now also selects which input DFB token the writer builds from |
| compute CTA 0/1 (`:349-350`) | `src1_cb_index`, `src2_cb_index`, read into the dead `[[maybe_unused]]` `in1_cb`/`in2_cb` | **dropped.** These are CB indices, and rule 2 of the kernel-side whitelist is explicit that a CB index becomes a DFB binding and *never* a named argument — but this kernel never touches either buffer, so there is no endpoint to declare and binding them would invent one. The scalar arg schema is unaffected: the dead **RTA** `is_input1` is still declared and still read (kept `[[maybe_unused]]`), so the host's per-node arg emission is unchanged. This is a deliberate, narrow departure from the brief's blanket *"the port does not remove them"*, which is written about dead **scalar** args; see `METAL2_PORT_REPORT.md` → *Open items* #2 |

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
- **Caution: Porting a shared kernel — rung 2 (create the fork), intra-op shape** — eleven `_metal2`
  forks beside their originals across the two passes (five in pass 1, six in pass 2), pointer comment
  added to each original, originals otherwise untouched.

Pass 2 adds these:

- **Aliased DFBs**, again — `TF_UNTILIZED_CACHE_DFB` (`c_24`) + `TF_UNTILIZED_CACHE2_DFB` (`c_25`) in
  the tiled fused factory, and `RMF_UNTILIZED_CACHE_DFB` (`c_5`) + `RMF_UNTILIZED_CACHE2_DFB` (`c_6`)
  in the row-major one. Same algorithm, same in-place-patch-and-republish shape, same non-negotiable
  "do not split".
- **Conditional / optional DFB and tensor bindings**, again — `INDEX` + `PAGE_TABLE` on both fused
  factories, gated by `USE_INDEX_TENSOR` / `IS_PAGED_CACHE` defines, plus a third gate
  `PAGE_TABLE_IS_DRAM` on the reader that decides whether `tensor::page_table` is bound at all.
- **Conditional `borrowed_from`** — the `INDEX` and `PAGE_TABLE` DFBs borrow the tensor's memory only
  when that tensor is L1-sharded, mirroring legacy's `nullptr`-or-`Buffer*` `.buffer` field. The same
  `DataflowBufferSpec` is a borrowed view on one path and a plain allocation on the other; nothing
  else about it changes.
- **Runtime-selected binding across two resources, on both channels** (audit Question #1) — the
  answer differs by channel, and the difference is forced by how the two token types carry identity:
  - *DFB channel:* `DFBBindingToken` holds its id in a runtime `uint16_t` member, so all DFB tokens
    share one type and a **ternary works**: `const DFBBindingToken t = is_input1 ? dfb::src1 :
    dfb::src2; DataflowBuffer dfb_input(t);`. Pure token substitution — no restructuring.
  - *Tensor channel:* host codegen emits a **distinct type per binding**
    (`TensorBindingToken<cta_offset, addr_crta_offset>`), and two bindings on one kernel necessarily
    occupy distinct slots, so the types always differ and a ternary can never compile. The shape is
    a **generic-lambda body instantiated twice**, selected once per invocation:
    `if (is_input1) { body(TensorAccessor(tensor::cache1)); } else { body(TensorAccessor(tensor::cache2)); }`.
    Inside the loop the accessor is concrete and fully inlined, so codegen matches legacy. The
    `KernelAdvancedOptions::TensorBindingSequence` alternative was checked and rejected: it works,
    but its type-erased wrapper turns each `get_noc_addr` into an indirect call, and both call sites
    are inside the per-tile loop.

---

## Deferred / Flagged

### New finding the audit missed — per-coord variation blocked the four `*MeshWorkloadFactory` factories (pass 1); the vehicle has since merged

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

**Resolution, found after this plan was first written:** the intended vehicle named at
`ttnn/api/ttnn/metal_v2_artifacts.hpp:20-22` — *"A future `MeshWorkloadSpecFactoryConcept` will
return a different (multi-program) artifact type"* — is implemented by
**Diego's mesh-workload branch, [PR #54988](https://github.com/tenstorrent/tt-metal/pull/54988)**, in review at the time of this port.
It adds `create_mesh_workload_artifacts(..., const MeshCoordinateRangeSet& tensor_coords)` returning
`MeshWorkloadArtifacts`, whose `programs` vector carries a `{range, spec, run_params}` entry per
coordinate range — giving per-coordinate specs *and* per-coordinate run args on the cache miss, with
range omission permitted (which is how the empty-descriptor idiom is expressed). All four mesh
factories are portable once it merges; the two factories ported in this pass need no change.
Note also the framing correction recorded in `METAL2_PORT_REPORT.md` → Handoff points #1:
`fill_cache`'s mesh factory is **not** multi-program — its spec is identical across coordinates and
only the `noop` run-arg value differs.

**Status at pass 2.** PR #54988 has **merged** — `9fb0ed54794` *"Let a spec factory build a different
program per mesh coordinate (#54988)"*, and `MeshWorkloadSpecFactoryConcept` /
`MeshWorkloadArtifacts::PerCoordProgram` are on this branch
(`ttnn/api/ttnn/operation_concepts.hpp:118-132`, `ttnn/api/ttnn/metal_v2_artifacts.hpp:38-52`). So the
framework gap this entry recorded is **closed**, and the four mesh factories are no longer blocked.

They are still not ported, for a different and narrower reason: their target concept is
`MeshWorkloadSpecFactoryConcept`, and **this recipe's procedure covers only
`ProgramSpecFactoryConcept` and `CustomProgramSpecFactoryConcept`** — it names this exact case and
rules that stopping is the correct outcome rather than improvising a multi-program port out of a
single-program procedure. See [Scope — two passes](#scope--two-passes) for the quoted wording from
both the port recipe and the audit-side doc, and `METAL2_PORT_REPORT.md` → *Handoff points* #1 for the
design work already done against the merged concept.

**Consequence for both passes:** the four mesh factories stay on `ProgramDescriptorFactoryConcept`, and
because each shares its kernel sources with the single-device sibling that *is* converting, all eleven
kernels those four pairs bind are forked per *Caution: Porting a shared kernel* rung 2. Recorded as a
Handoff point in `METAL2_PORT_REPORT.md`.

### Closed in pass 2 — audit Question #1, both channels

The audit's open design question ("How should the fused factories' runtime-selected input DFB be
expressed in a `ProgramSpec`?") was answered on both channels at the end of pass 1, satisfying the
brief's instruction *"Get an answer before you write the fused specs."* **Pass 2 implements that
answer**; the record below is the design as resolved, and
[Applied Patterns](#applied-patterns) carries the realized shape.

**DFB channel — resolved.** Bind both `src1` and `src2` to every fused `KernelSpec` unconditionally and
let the kernel select which binding token to construct its `DataflowBuffer` from, using the existing
`is_input1` runtime arg (`const DFBBindingToken input_dfb = is_input1 ? dfb::src1 : dfb::src2;`).
Kernel-side this is a pure token substitution — the tiled compute kernel already branches at runtime
into two compile-time instantiations, so it needs no restructuring at all. Host-side it is legal
(1 PRODUCER + 1 CONSUMER per node for each DFB) and **free**: a borrowed DFB skips L1 allocation
(`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2519-2521`), so the widened derived placement costs
no L1, and both borrowed-DFB validation checks are whole-buffer rather than per-node. Critically,
`is_input1` stays an RTA — no promotion to a `#define`, no arg-schema change. The one delta from legacy
(each borrowed DFB configured on the half of `all_cores_bb` where legacy left it unconfigured) is inert
on Gen1 and recorded as Quasar-uplift debt.

**Tensor channel — resolved.** Reader RTA[2] / writer RTA[1] carry `cache_tensor1` on `cores1` and
`cache_tensor2` on `cores2`. Neither tensor is optional (both are plain `Tensor` members of
`PagedFusedUpdateCacheInputs`), so this is not a conditional-binding problem. A `TensorBinding`'s base
address is delivered as an implicit **CRTA** (broadcast to every node), so binding both puts both
addresses on every core — already correct everywhere, with **no placement delta and no Quasar debt**,
unlike the DFB half. The ternary does not transfer, because codegen emits a distinct *type* per
binding (`TensorBindingToken<cta_offset, addr_crta_offset>`, and two bindings necessarily occupy
distinct slots). Resolution: **branch on `is_input1` with two typed accessors**, via a generic lambda
so the loop body is written once. The `AbstractTensorAccessorWrapper` alternative was verified to work
in both NoC directions (`noc_traits.h:140-171` specializes `src_addr` *and* `dst_addr`) but costs an
un-devirtualizable indirect call per page inside the per-tile loop, so the branch is preferred. Full
analysis in `METAL2_PORT_REPORT.md` → *Open items for downstream* #1.

**Dependency note:** these two single-device factories needed **no framework change** and were ported
in pass 2 against `main`. Their `*MeshWorkloadFactory` siblings now have their framework vehicle
(#54988, merged) but await a port procedure for `MeshWorkloadSpecFactoryConcept`, per the finding
above.

### Other flags

- **No `TensorParameter` relaxation is declared anywhere.** The audit reports `none` on all rows; the
  port keeps strict matching.
- **No varargs.** Every ported kernel reaches each argument as a distinct field a fixed number of
  times, so every argument is named. Neither `num_runtime_varargs` nor `compile_time_varargs` is used.
- **No Case 2 tensor binding**, so no `get_bank_base_address` bridge and no compute-kernel raw-pointer
  block.
- **No `GlobalCircularBuffer`** anywhere in the op.
