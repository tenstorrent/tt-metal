# Port Plan — `experimental/paged_cache`

Port plan for `ttnn/cpp/ttnn/operations/experimental/paged_cache`, ported from `ProgramDescriptor`
to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.

Three DeviceOperations, eight factories, eleven kernels. The four single-device factories land on
`CustomProgramSpecFactoryConcept`; the four `*MeshWorkloadFactory` variants land on
`MeshWorkloadSpecFactoryConcept` (see [TTNN ProgramFactory](#ttnn-programfactory) for why the brief's
concept does not fit those four, and the authorisation to proceed).

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` on all eight factories — each defines `create_descriptor`
  returning `tt::tt_metal::ProgramDescriptor`. No `create_workload_descriptor` anywhere.
- Factory methods live in factory structs (a `program_factory_t` variant on each device-op), not
  directly on the device-operation. Exception 3 (direct-descriptor) does **not** apply.
- Variants per device-op:
  - `PagedFillCacheDeviceOperation` → `std::variant<PagedFillCacheProgramFactory, PagedFillCacheMeshWorkloadFactory>`
  - `PagedUpdateCacheDeviceOperation` → `std::variant<PagedUpdateCacheProgramFactory, PagedUpdateCacheMeshWorkloadFactory>`
  - `PagedFusedUpdateCacheDeviceOperation` → `std::variant<` tiled + tiled-mesh + row-major + row-major-mesh `>`
- `select_program_factory` picks the mesh variant **iff** `operation_attributes.mesh_coords.has_value()`
  ([fill_cache](device/fill_cache/paged_fill_cache_device_operation.cpp#L17-L24)), and the factory index
  is folded into the program hash — so the plain and mesh variants never share a cache entry.
- Custom `compute_program_hash`: present on **all three** device-ops. **Left intact; not touched.**
  - [`device/fill_cache/paged_fill_cache_device_operation.cpp:216-225`](device/fill_cache/paged_fill_cache_device_operation.cpp#L216-L225)
  - [`device/update_cache/paged_update_cache_device_operation.cpp:314-338`](device/update_cache/paged_update_cache_device_operation.cpp#L314-L338)
  - [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:250-268`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L250-L268)

  Each hash deliberately excludes scalar attributes (`update_idxs`, `batch_offset`,
  `batch_idx_fallback`, `noop`) that the matching `override_runtime_arguments` re-applies on every
  cache hit. The exclusion and the re-application are two halves of one design; the translated
  override re-applies the same set.
- No backdoor `attribute_values` / `to_hash`. No `get_dynamic_runtime_args`.

---

### Variant: `PagedFillCacheProgramFactory`

Built by the anonymous-namespace helper `build_paged_fill_cache_descriptor`, shared with the mesh
variant; the two differ only in the `noop` value.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_fill_cache_interleaved.cpp` | `all_cores` | 0 `src0_cb_index`(c_0), 1 `Wt`, 2.. `TensorAccessorArgs(src)` | 0 `src_buffer`\*, 1 `start_tile_id`, 2 `num_rows`, 3 `noop` | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_fill_cache_interleaved.cpp` | `all_cores` | 0 `src0_cb_index`(c_0), 1 `page_table_cb_index`(c_1), 2 `num_heads`, 3 `num_blocks_of_work_per_head`, 4 `block_size_t`, 5 `Wt`, 6 `log2_page_table_stick_size_B`, 7 `page_table_stick_size_B`, 8 `use_batch_idx_tensor`, 9 `cb_batch_idx_id`(c_2), 10 `batch_idx_stick_size_B`, 11 `batch_idx_num_elements`, 12 `num_blocks_of_work_per_batch`, 13 `capacity_t`, 14 `use_valid_seq_len`, 15 `cb_valid_seq_len_id`(c_3), 16 `valid_seq_len_stick_size_B`, 17.. 4×`TensorAccessorArgs` (dst, page_table, batch_idx, valid_seq_len) | 0 `dst_buffer`\*, 1 `page_table_buffer`\*, 2 `start_row_num`, 3 `num_rows`, 4 `batch_idx_tensor buffer`\* **or** `batch_idx_fallback` scalar, 5 `noop`, 6 `valid_seq_len buffer`\* **or** literal `0` | none | O2 (unset, DM) | `WriterConfigDescriptor{}` |

No compute kernel. (\* = `Buffer*` pushed via `emplace_runtime_args`.)

#### CBs

| index | total_size | core_ranges | data_format | page_size | condition |
|---|---|---|---|---|---|
| c_0 input tiles | `Wt*2 * single_tile_size` | `all_cores` | input dtype | `single_tile_size` | always |
| c_1 page table | `page_table_stick_size_B` | `all_cores` | page_table dtype | `page_table_stick_size_B` | always |
| c_2 batch_idx | `batch_idx_stick_size_B * batch_idx_num_elements` | `all_cores` | batch_idx dtype | `batch_idx_stick_size_B` | `use_batch_idx_tensor` |
| c_3 valid_seq_len | `valid_seq_len_stick_size_B` | `all_cores` | UInt32 | `valid_seq_len_stick_size_B` | `use_valid_seq_len` |

No `.buffer` on any of the four — no borrowed memory here. No aliasing (all single-element
`format_descriptors`).

#### Semaphores

none

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| [`reader_fill_cache_interleaved.cpp:29`](device/kernels/dataflow/reader_fill_cache_interleaved.cpp#L29) | `input_tensor` | reader 0 |
| [`writer_fill_cache_interleaved.cpp:145`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L145) | `cache_tensor` (in-place output) | writer 0 |
| [`writer_fill_cache_interleaved.cpp:146`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L146) | `page_table` | writer 1 |
| [`writer_fill_cache_interleaved.cpp:101`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L101) | `batch_idx_tensor_opt` | writer 4 (optional) |
| [`writer_fill_cache_interleaved.cpp:122`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L122) | `valid_seq_len_tensor_opt` | writer 6 (optional) |

#### Work split

- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major=true)`
- Per-core counts ride as **RTAs** (`start_tile_id` / `start_row_num`, `num_rows`), not per-group CTAs —
  so there is **one** `KernelDescriptor` per kernel over `all_cores`, and no multiplicity to preserve.

---

### Variant: `PagedUpdateCacheProgramFactory`

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `.../dataflow/reader_update_cache_interleaved_start_id.cpp` | `all_cores` | 0 c_0, 1 c_1, 2 `use_index_tensor`, 3 c_2, 4 `cache_batch_num_tiles`, 5 `Wt`, 6 `log2_page_size`(dead, always 0), 7 `index_stick_size`, 8 `is_paged_cache`, 9 `num_heads`, 10 `block_size`, 11 `block_size_t`, 12 `max_blocks_per_seq`(dead), 13 `log2_page_table_stick_size`(dead), 14 `page_table_stick_size`, 15 c_3, 16 `St`, 17 `sem_id`, 18 `cache_position_modulo`, 19.. 3×TAArgs (cache, index, page_table) | 0 `dst_buffer`\*, 1 `cache_start_id`, 2 `index buffer`\* or 0, 3 `i` (batch idx), 4 `page_table buffer`\* or 0, 5 `wait_to_start` | O2 | `ReaderConfigDescriptor{}` |
| writer | `.../dataflow/writer_update_cache_interleaved_start_id.cpp` | `all_cores` | 0 c_16 (**named `cache_cb_id` in-kernel but wired to the *output* CB**), 1 c_24, 2 c_25, 3 c_26, 4 `use_index_tensor`, 5 c_2, 6 `cache_batch_num_tiles`, 7 `Wt`, 8 `Wbytes`, 9 `is_paged_cache`, 10 `num_heads`, 11 `block_size`, 12 `block_size_t`, 13 `max_blocks_per_seq`(dead), 14 c_3, 15 `St`, 16 `sem_id`, 17 `cache_position_modulo`, 18.. TAArgs (cache) | 0 `dst_buffer`\*, 1 `cache_start_id`, 2 `tile_update_offset_B`, 3 `i`, 4 `send_signal`, 5 `send_core_x`, 6 `send_core_y` | O2 | `WriterConfigDescriptor{}` |
| compute | `.../compute/update_cache.cpp` | `all_cores` | 0 c_0, 1 c_1, 2 c_24, 3 c_25, 4 c_26, 5 c_16, 6 `Wt`, 7 `num_heads` | none | **O3** (unset on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | notes |
|---|---|---|---|---|---|
| c_0 cache tiles | `2*Wt * cache_single_tile_size` | `all_cores` | cache dtype | cache tile | |
| c_1 input shard | `num_input_tiles * input_single_tile_size` | `all_cores` | input dtype | input tile | **`.buffer = in1_buffer`** → borrowed memory |
| c_24 + c_25 | `2*Wt * interm_single_tile_size` | `all_cores` | interm | interm tile | **one `CBDescriptor`, two `format_descriptors` → ALIASED** |
| c_26 | `2*Wt * interm_single_tile_size` | `all_cores` | interm | interm tile | |
| c_16 output | `B*Wt * cache_single_tile_size` | `all_cores` | cache dtype | cache tile | |
| c_2 index | `index_tensor_tile_size` | `all_cores` | index dtype | index tile | `use_index_tensor` only; **no** `.buffer` here |
| c_3 page table | `page_table_stick_size` | `all_cores` | page_table dtype | stick | `is_paged_cache` only; no `.buffer` |

`interm_cb_data_format = fp32_dest_acc_en ? Float32 : Float16_b`.

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`in0_sequential_mode`) | WORKER | `all_cores` | 0 |

Allocated **unconditionally**, even when `share_cache` is false and no kernel touches it. Preserved as-is.

#### Work split

- Driver: the input tensor's **shard grid** — `all_cores = input_tensor.shard_spec()->grid`;
  `cores = corerange_to_cores(grid, grid.num_cores(), row_major)`. One core per batch entry.
- Single `KernelDescriptor` per kernel over `all_cores`; per-core variation is entirely RTA. No multiplicity.

---

### Variant: `PagedTiledFusedUpdateCacheProgramFactory`

#### Kernels

Three kernels, all over **`all_cores_bb`** — the bounding box of the two input shard grids.

| unique_id | source | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|
| reader | `.../dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp` | 0 c_1, 1 c_2, 2 c_0, 3 `use_index_tensor`, 4 `index_is_dram`, 5 c_3, 6 `cache_batch_num_tiles`, 7 `Wt`, 8 `log2_page_size`(dead), 9 `index_stick_size`, 10 `is_paged_cache`, 11 `num_heads`, 12 `block_size`, 13 `block_size_t`, 14 `max_blocks_per_seq`(dead), 15 `log2_page_table_stick_size`(dead), 16 `page_table_stick_size`, 17 `page_table_is_dram`, 18 c_4, 19 `St`, 20 `sem_id`, 21 `B`, 22.. 3×TAArgs | 0 `has_work`, 1 `is_input1`, 2 `dst{1,2}_buffer`\*, 3 `cache_start_id`, 4 `index`\* or 0, 5 `i`, 6 `page_table`\* or 0, 7 `wait_to_start` | O2 | `ReaderConfigDescriptor{}` |
| writer | `.../dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp` | 0 c_16, 1 c_24, 2 c_25, 3 c_26, 4 `use_index_tensor`, 5 c_3, 6 `cache_batch_num_tiles`, 7 `Wt`, 8 `Wbytes`, 9 `is_paged_cache`, 10 `num_heads`, 11 `block_size`, 12 `block_size_t`, 13 `max_blocks_per_seq`(dead), 14 c_4, 15 `St`, 16 `sem_id`, 17 `B`, 18 `page_table_stick_size`, 19 `page_table_is_dram`, 20.. TAArgs | 0 `has_work`, 1 `dst{1,2}`\*, 2 `cache_start_id`, 3 `tile_update_offset_B`, 4 `i`, 5 `send_signal`, 6 `send_core_x`, 7 `send_core_y` | O2 | `WriterConfigDescriptor{}` |
| compute | `.../compute/paged_fused_update_cache.cpp` | 0 c_1, 1 c_2, 2 c_0, 3 c_24, 4 c_25, 5 c_26, 6 c_16, 7 `Wt`, 8 `num_heads` | 0 `has_work`, 1 `is_input1` | **O3** | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en}` |

Cores in `unused_cores` (in the bounding box, in neither shard grid) get a **one-element** RTA vector
`{!has_work}` on all three kernels — one `KernelDescriptor` carrying two different arg-vector lengths.

#### CBs

| index | core_ranges | notes |
|---|---|---|
| c_0 cache tiles | `all_cores_bb` | |
| c_1 input1 shard | **`input1_cores`** | `.buffer = in1_buffer` → borrowed |
| c_2 input2 shard | **`input2_cores`** | `.buffer = in2_buffer` → borrowed |
| c_24 + c_25 | `all_cores_bb` | **one `CBDescriptor`, two `format_descriptors` → ALIASED** |
| c_26 untilized input | `all_cores_bb` | |
| c_16 output | `all_cores_bb` | |
| c_3 index | `all_cores_bb` | `use_index_tensor`; `.buffer = index_buffer_ptr` — **non-null only when the index tensor is L1-sharded** |
| c_4 page table | `all_cores_bb` | `is_paged_cache`; `.buffer = page_table_buffer_ptr` — **non-null only when the page table is L1-sharded** |

#### Semaphores

One (`in0_sequential_mode`), WORKER, `all_cores_bb`, initial 0. Unconditional.

#### Work split

- `input1_cores` / `input2_cores` from the two input tensors' shard grids; asserted disjoint and
  equal-size by the device-op
  ([`paged_fused_update_cache_device_operation.cpp:229-235`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L229-L235)).
- `cores1[i]` and `cores2[i]` both handle batch entry `i`.

---

### Variant: `PagedRowMajorFusedUpdateCacheProgramFactory`

Structurally identical to the tiled factory except:

- **No `untilized_input` CB.** CB indices are c_0 cache, c_1/c_2 inputs, c_3 index, c_4 page table,
  **c_5** intermed0, **c_6** intermed1, **c_7** output. c_5+c_6 are again **one aliased `CBDescriptor`**.
- The **writer**, not compute, drains the input CB: its CTAs 3 and 4 are `untilized_input1_cb_id` (c_1)
  and `untilized_input2_cb_id` (c_2), runtime-selected by an extra `is_input1` **RTA** (writer RTA 8).
- The **compute kernel never touches an input CB.** Its CTAs 0/1 (`in1_cb`, `in2_cb`) and its `is_input1`
  RTA feed a `[[maybe_unused]]` local that is never read
  ([`compute/paged_row_major_fused_update_cache.cpp:19-25`](device/kernels/compute/paged_row_major_fused_update_cache.cpp#L19-L25)).
- The reader pushes **1** entry to the input DFB (not `Wt`).

### Shared kernels

**None.** All 11 kernel sources live in this op's `device/kernels/` tree; a repo-wide grep for
`paged_cache/device/kernels` finds referrers only inside this op directory, and no `_metal2` fork
exists beside any of them.

**No fork is created by this port.** The shared-kernel Caution applies to a source bound by factories
that will **not all convert in the same change** — and here every binder of every kernel converts in
this change: each mesh factory delegates its program build to its single-device sibling, so the pair
converts as a unit, and no two device-ops share a kernel. Converting each source in place is therefore
correct and leaves no legacy binder behind. (The audit brief's "this port creates the first fork for
each" reads the rung-1 check — *does a fork already exist?* — as though it implied rung 2. It does not;
rung 2 is only reached when a binder is left behind. Recorded as friction in the port report.)

### Flags

- No unreferenced kernel files.
- **Two aliased `CBDescriptor`s the audit's CB census did not flag** — `update_cache` c_24+c_25, tiled
  fused c_24+c_25, row-major fused c_5+c_6 each place **two** `buffer_index`es on one `CBDescriptor`.
  The census lists them as independent CBs. They need `advanced_options.alias_with`; see
  [Applied Patterns](#applied-patterns).
- **Ragged per-core RTA vectors** in both fused factories (8- or 9-element on working cores, 1-element
  on `unused_cores`, from one `KernelDescriptor`). Dissolved by the WorkUnitSpec split.

---

## TTNN ProgramFactory

- **Concept (single-device factories, 4 of 8): `CustomProgramSpecFactoryConcept`** — inherited from the
  brief. Each has an `override_runtime_arguments`, which selects the custom concept over the base one.
- **Concept (mesh-workload factories, 4 of 8): `MeshWorkloadSpecFactoryConcept`** — **a deliberate
  departure from the brief**, made on explicit invoker authorisation. Rationale below.
- **Custom `compute_program_hash`**: present on all three device-ops (sites above). Left intact.
- **Pybound `create_descriptor`**: none. [`paged_cache_nanobind.cpp:23-165`](paged_cache_nanobind.cpp#L23-L165)
  binds only the three user-facing op functions, so the port removes no pybind surface and makes no
  user-visible API change.

### Why the four mesh factories cannot take the brief's concept

The brief names `CustomProgramSpecFactoryConcept` for all eight rows. Four of them build a
**per-mesh-coordinate** program, and that concept has no channel for a per-coordinate value in
*either* direction:

- **Cache miss.** The adapter calls `create_program_artifacts` **once** — it takes no coordinate — and
  applies the *same* `ProgramRunArgs` to every range
  ([`mesh_device_operation_adapter.hpp:971-985`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L971-L985)).
  The adapter says so itself at
  [`:1005-1007`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1005-L1007):
  *"per-coordinate run args must come from a MeshWorkloadSpecFactoryConcept factory."*
- **Cache hit.** `override_runtime_arguments` is called once per **range**, not per coordinate
  ([`:1017-1025`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1017-L1025)). With
  uniform tensor storage the whole mesh collapses to **one** range
  ([`device_operation.hpp:334-336`](../../../../../../ttnn/api/ttnn/device_operation.hpp#L334-L336)), so
  the override runs once with coordinate `(0,0)` and its result applies to every chip.

The legacy descriptor adapter, by contrast, iterates `tensor_coords.coords()` and builds **one program
per coordinate** when `create_descriptor` takes one
([`:607-612`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L607-L612)).

Porting `PagedFillCacheMeshWorkloadFactory` onto the custom concept would therefore give every chip the
`noop` computed for coordinate `(0,0)` — i.e. **excluded chips would write to the cache**. The brief's
claim that this factory "is already in the recommended shape" because its exclusion rides on a runtime
arg does not hold: there is no per-coordinate `ProgramRunArgs` on that concept. The brief's recommended
answer for the other three (build the full program, set an all-cores `has_work = 0`) fails for the same
reason.

`MeshWorkloadSpecFactoryConcept` is the concept that expresses this, and it **is** implemented in this
tree ([`operation_concepts.hpp:118-132`](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L118-L132),
adapter at [`mesh_device_operation_adapter.hpp:1035-1130`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1035-L1130)).
The port recipe covers only the two single-program spec concepts and directs a porter to stop when the
target is a mesh-workload concept. What was actually hit here is the *porter-disagrees-with-the-audit*
case, whose rule is to surface it to the invoker — which was done, and the invoker authorised
proceeding onto `MeshWorkloadSpecFactoryConcept` in this pass.

### Implementation notes

- Each mesh factory's `create_mesh_workload_artifacts` calls its single-device sibling's
  `create_program_artifacts` and stamps the resulting spec across the mesh. No new spec construction:
  the `ProgramSpec` genuinely does not vary by coordinate for any of these ops — what varies is *which*
  coordinates get a program (update_cache, both fused) or *one run-arg value* (fill_cache's `noop`).
- Granularity is **one program per coordinate**, matching what the ported-from descriptor adapter built.
  That reproduces the ported-from program set exactly and makes each range uniform in `noop`.
- The two mesh-filtering idioms are preserved as they are: `update_cache` and both fused **omit the
  range** for an excluded coordinate; `fill_cache` emits every coordinate and patches only `noop`.

---

## Planned Spec Shape

### `PagedFillCacheProgramFactory`

- **KernelSpecs** (2): `READER`, `WRITER`.
- **DataflowBufferSpecs** (2 + 2 conditional): `IN_TILES`(c_0), `PAGE_TABLE`(c_1),
  `BATCH_IDX`(c_2, iff `use_batch_idx_tensor`), `VALID_SEQ_LEN`(c_3, iff `use_valid_seq_len`).
- **SemaphoreSpecs**: none.
- **TensorParameters** (3 + 2 conditional): `INPUT`, `CACHE`, `PAGE_TABLE_T`,
  `BATCH_IDX_T`(cond), `VALID_SEQ_LEN_T`(cond).
- **WorkUnitSpecs** (1): `main` over `all_cores`, kernels {READER, WRITER}.

### `PagedUpdateCacheProgramFactory`

- **KernelSpecs** (3): `READER`, `WRITER`, `COMPUTE`.
- **DataflowBufferSpecs** (6 + 2 conditional): `CACHE_TILES`(c_0), `INPUT_SHARD`(c_1, borrowed),
  `UNTILIZED_CACHE`(c_24), `UNTILIZED_CACHE2`(c_25), `UNTILIZED_INPUT`(c_26), `OUT_TILES`(c_16),
  `INDEX`(c_2, cond), `PAGE_TABLE`(c_3, cond).
  `UNTILIZED_CACHE` ↔ `UNTILIZED_CACHE2` mutually aliased.
- **SemaphoreSpecs** (1): `IN0_SEQUENTIAL` over `all_cores`.
- **TensorParameters** (2 + 2 conditional): `CACHE`, `INPUT` (borrow-only — no kernel binds it),
  `INDEX_T`(cond), `PAGE_TABLE_T`(cond).
- **WorkUnitSpecs** (1): `main` over `all_cores`, kernels {READER, WRITER, COMPUTE}.

### `PagedTiledFusedUpdateCacheProgramFactory` — **two work units**

- **KernelSpecs** (6): `READER1`/`WRITER1`/`COMPUTE1` and `READER2`/`WRITER2`/`COMPUTE2` — same three
  sources, one triple per input shard grid.
- **DataflowBufferSpecs** (6 + 2 conditional): `CACHE_TILES`(c_0), `INPUT1`(c_1, borrowed),
  `INPUT2`(c_2, borrowed), `UNTILIZED_CACHE`(c_24) ↔ `UNTILIZED_CACHE2`(c_25) aliased,
  `UNTILIZED_INPUT`(c_26), `OUT_TILES`(c_16), `INDEX`(c_3, cond), `PAGE_TABLE`(c_4, cond).
- **SemaphoreSpecs** (1): `IN0_SEQUENTIAL` over `all_cores` (the **union**, not the bounding box).
- **TensorParameters** (4 + 2 conditional): `CACHE1`, `CACHE2`, `INPUT1`, `INPUT2` (borrow-only),
  `INDEX_T`(cond), `PAGE_TABLE_T`(cond).
- **WorkUnitSpecs** (2): `wu_input1` over `input1_cores` {READER1, WRITER1, COMPUTE1};
  `wu_input2` over `input2_cores` {READER2, WRITER2, COMPUTE2}.

### `PagedRowMajorFusedUpdateCacheProgramFactory` — **two work units**

As tiled, minus `UNTILIZED_INPUT`, with intermed/output at c_5/c_6/c_7, and with the **writer**
(not compute) binding the input DFB.

---

## Preserved Multiplicity

The two fused factories gain same-source `KernelSpec` multiplicity that the legacy code did **not**
have — it is created by the port, not preserved from legacy, and it is *forced*:

```
Legacy KernelDescriptor [reader] of reader_paged_fused_update_cache_interleaved_start_id.cpp
                        over all_cores_bb, picking c_1 vs c_2 from the `is_input1` RTA
  -> KernelSpecs [READER1, READER2] of the same source
  -> in WorkUnitSpecs [wu_input1 (input1_cores), wu_input2 (input2_cores)]
  -> READER1 binds INPUT1 PRODUCER + CACHE_TILES PRODUCER; READER2 binds INPUT2 PRODUCER + CACHE_TILES PRODUCER
```

…and identically for the writer and compute sources, in both fused factories.

**Why forced.** A `dfb::` token is a compile-time name, and a DFB has **no core range of its own** — its
node set is derived from the union of its bound kernels' `WorkUnitSpec::target_nodes`
([`dataflow_buffer_spec.hpp:57-59`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp#L57-L59)).
One `KernelSpec` over `all_cores_bb` binding both input DFBs would place `c_1` on `input2_cores` nodes
too, where the `borrowed_from` `input_tensor1` has no resident shard. Narrowing each `WorkUnitSpec` to
one shard grid is the only expressible placement.

The two `WorkUnitSpec`s cover **disjoint** node sets (asserted by the device-op), so each node sees
exactly one instance of each source and every shared DFB is an ordinary 1P+1C there — this is the
disjoint-node work-split, **not** the same-grid two-toucher case, and **not** the multi-binding flag.

`fill_cache` and `update_cache`: **none** — no work-split multiplicity, in legacy or in the port.

### Consequence: the `unused_cores` instances drop out

`unused_cores` (inside the bounding box, in neither shard grid) falls outside both `WorkUnitSpec`s, so
those cores get no kernels and no DFB reservation. **No output value moves** — they early-exit today on
`has_work = 0` and are simply absent afterwards. The op dispatches to fewer cores and stops reserving
roughly **59–112 KB on each**. The set is empty in the unit tests but holds **20 cores** in the
llama3-70b galaxy decode configuration. Observable only as a profiler trace with fewer cores and an
SRAM report with less reserved; recorded in the port report.

---

## Dropped Plumbing

### `PagedFillCacheProgramFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA 0 | `src0_cb_index` | `DFBBinding(IN_TILES, "in", PRODUCER)` |
| reader CTA 2.. | `TensorAccessorArgs(src_buffer)` | `TensorBinding(INPUT, "input")` |
| reader RTA 0 | `src_buffer` (`Buffer*`) | `TensorBinding(INPUT, …)` |
| writer CTA 0 | `src0_cb_index` | `DFBBinding(IN_TILES, "in", CONSUMER)` |
| writer CTA 1 | `page_table_cb_index` | `DFBBinding(PAGE_TABLE, "page_table", PRODUCER+CONSUMER)` |
| writer CTA 8 | `use_batch_idx_tensor` | `#define USE_BATCH_IDX_TENSOR` |
| writer CTA 9 | `cb_batch_idx_id` | `DFBBinding(BATCH_IDX, …)` (conditional) |
| writer CTA 14 | `use_valid_seq_len` | `#define USE_VALID_SEQ_LEN` |
| writer CTA 15 | `cb_valid_seq_len_id` | `DFBBinding(VALID_SEQ_LEN, …)` (conditional) |
| writer CTA 17.. | 4× `TensorAccessorArgs` | 4× `TensorBinding` (2 conditional) |
| writer RTA 0, 1 | `dst_buffer`, `page_table_buffer` | `TensorBinding(CACHE)`, `TensorBinding(PAGE_TABLE_T)` |
| writer RTA 4 | `batch_idx_tensor->buffer()` **or** `batch_idx_fallback` | `TensorBinding(BATCH_IDX_T)` when present; **named RTA `batch_idx_fallback`** otherwise |
| writer RTA 6 | `valid_seq_len->buffer()` **or** literal `0` | `TensorBinding(VALID_SEQ_LEN_T)` when present; **dropped entirely** otherwise (the kernel's read is dead under `#ifndef USE_VALID_SEQ_LEN`) |
| all remaining positional CTAs/RTAs | positional | named (`Wt`, `num_heads`, `start_tile_id`, `num_rows`, `noop`, …) |

### `PagedUpdateCacheProgramFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTAs 0,1,3,15 · writer CTAs 0,1,2,3,5,14 · compute CTAs 0-5 | CB indices | `DFBBinding`s |
| reader CTA 2 / writer CTA 4 | `use_index_tensor` | `#define USE_INDEX_TENSOR` (emitted to **reader and writer**) |
| reader CTA 8 / writer CTA 9 | `is_paged_cache` | `#define IS_PAGED_CACHE` (emitted to **reader and writer**) |
| reader CTA 17 / writer CTA 16 | `in0_sequential_mode_semaphore_id` | `SemaphoreBinding(IN0_SEQUENTIAL, "in0_seq")` |
| reader CTA 19.. / writer CTA 18.. | `TensorAccessorArgs` | `TensorBinding`s |
| reader RTA 0, 2, 4 · writer RTA 0 | `Buffer*` | `TensorBinding`s |
| — | `.buffer = in1_buffer` on the c_1 CB | `DataflowBufferSpec::borrowed_from = INPUT` |
| — | `UpdateDynamicCircularBufferAddress` in the override | the borrowed DFB re-resolves from its `TensorArgument`; no `dfb_run_overrides` entry needed |

### Both fused factories

As `update_cache`, plus:

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTAs 0,1 (tiled + RM) · compute CTAs 0,1 (tiled) · writer CTAs 3,4 (RM) | `src1_cb_index`, `src2_cb_index` — runtime-selected | **one** `DFBBinding` per `KernelSpec`, chosen structurally by work unit |
| reader RTA 1 · compute RTA 1 (tiled) · writer RTA 8 (RM) | `is_input1` | **dropped** — dissolved into the two-`WorkUnitSpec` structure |
| reader CTA 4 | `index_is_dram` | `#define INDEX_IS_DRAM` |
| reader CTA 17 / writer CTA 19 | `page_table_is_dram` | `#define PAGE_TABLE_IS_DRAM` |
| — | `.buffer = index_buffer_ptr` (L1-sharded only) | `borrowed_from = INDEX_T`, only in the sharded config |
| — | `.buffer = page_table_buffer_ptr` (L1-sharded only) | `borrowed_from = PAGE_TABLE_T`, only in the sharded config |
| RM compute CTAs 0,1 | `in1_cb`, `in2_cb` feeding a `[[maybe_unused]]` local | **dropped** — see below |
| `unused_cores` RTA vectors `{!has_work}` | ragged per-core RTA | **dropped** — those nodes are outside both work units |

**`has_work` is kept** as a named RTA, set to `1` on every node of both work units. The kernels' early
`return` is preserved verbatim; only the ragged one-element vectors on `unused_cores` go away.

**Dropping the row-major compute kernel's two dead CB-index CTAs is forced, not tidying.** A CB-index
CTA converts to a `DFBBinding` (kernel-side whitelist rule 2), and there is no binding to make: that
kernel never touches an input DFB. Binding one anyway would add a third toucher to a DFB whose census
is exactly reader-P + writer-C, forcing the multi-binding flag for a buffer nothing reads. The value
was never read, so dropping it is zero-functional-change. The op's other dead args are **scalars**
(`max_blocks_per_seq`, `log_base_2_of_page_size`, `log2_page_table_stick_size`) and are **kept** as
named CTAs.

---

## Applied Patterns

- [Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings)
  — the largest single piece of work. Eight conditional DFBs and six conditional tensors across the four
  factories. Every kernel declares its `CircularBuffer` objects **unconditionally at function scope** and
  gates only the *use* behind `if constexpr`, which still performs name lookup on the discarded branch —
  so each gate is promoted to an `#ifdef` fed from `KernelSpec::compiler_options.defines`. The define
  must reach **every** kernel that names the resource, not just the one the legacy factory sent a flag to.
- [Aliased DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
  — the three two-element `CBDescriptor`s (`update_cache` c_24+c_25, tiled c_24+c_25, row-major c_5+c_6)
  become pairs of specs with mutual `advanced_options.alias_with`. All three legality rules hold: equal
  `num_entries * entry_size`, same bound kernel set {compute, writer}, neither member borrowed.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — `fill_cache`'s three metadata DFBs (`PAGE_TABLE`, `BATCH_IDX`, `VALID_SEQ_LEN`). Census re-derived
  and confirmed: the **writer alone** touches each (`reserve_back(1)`, `get_write_ptr()`, read back
  through an L1 pointer, never `push_back`). One toucher → bind the writer PRODUCER **and** CONSUMER.
- [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  (applied in reverse) — the two-`WorkUnitSpec` split of both fused factories, per *Preserved Multiplicity*.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — `compute_kernel_lib::untilize` / `tilize` take their handles as `uint32_t` non-type template
  parameters; `dfb::name`'s `constexpr` conversion covers template-parameter position. Also
  `compute_kernel_hw_startup(dfb::…, dfb::…)`.
- [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
  — the fused device-op selects tiled vs row-major; unchanged, each remains its own factory.
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  — the two fused factory `.cpp`s already sit behind `CMAKE_UNIQUE_NAMESPACE_*` guards; the port keeps
  its new spec-name constants inside those per-file anonymous namespaces.

**Not applied, and why:** the multi-binding advanced option is needed **nowhere** — every DFB's census
fits 1P+1C or (for `fill_cache`'s three metadata buffers) a one-toucher self-loop. No DFB is both
self-looped and multi-bound. No dead CB (zero endpoints) exists: the one config that could have
produced one, `is_paged_cache && !use_index_tensor`, is ruled out by device-op validation
([`paged_update_cache_device_operation.cpp:149`](device/update_cache/paged_update_cache_device_operation.cpp#L149)).

---

## Deferred / Flagged

- **The audit's target concept is wrong for 4 of 8 factories** — see
  [TTNN ProgramFactory](#why-the-four-mesh-factories-cannot-take-the-brief-s-concept). Surfaced to the
  invoker before any code was written; authorised.
- **Three aliased `CBDescriptor`s the audit's CB census missed** — see [Flags](#flags).
- **`MakeMeshWorkloadFromSpecs` requires at least one `ProgramSpec`**
  ([`program_spec.cpp:3385`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L3385)). On
  the three omit-the-range mesh factories, an empty or fully `tensor_coords`-disjoint `mesh_coords`
  therefore raises where the ported-from path silently dispatched nothing. Behaviour delta to confirm
  by test, not to fix; routed to the port report.
- `paged_fill_cache` accepts a `compute_kernel_config` and discards it
  ([`paged_cache.cpp:80-81`](paged_cache.cpp#L80-L81)); the fill_cache factory builds no compute kernel.
  Not touched.
- Both fused factories and `update_cache` resolve a full TTNN `ComputeKernelConfig` but copy only
  `fp32_dest_acc_en` onto the `ComputeConfigDescriptor`, silently ignoring `math_fidelity`,
  `math_approx_mode` and `dst_full_sync_en`. Preserved exactly (a `ComputeGen1Config` with only
  `enable_32_bit_dest` set reproduces it byte-for-byte, since the two structs' defaults coincide);
  reported, not repaired.
