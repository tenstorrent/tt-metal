# Port Plan — `data_movement/tilize_with_val_padding`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding`, ported from the
`ProgramDescriptor` (`create_descriptor`) API to Metal 2.0 (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass — three of the four factories.** `TilizeWithValPaddingSingleCoreFactory`,
`TilizeWithValPaddingMultiCoreDefaultFactory` and `TilizeWithValPaddingMultiCoreShardedFactory` are
ported. `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` is **deliberately excluded by the
invoker** — it cleared the audit but the op owner reports it is subtly broken, so it is to be left on
the legacy `ProgramDescriptorFactoryConcept` until that is resolved. This is legal: `program_factory_t`
may hold factories on different concepts and the framework dispatches per-factory
(`AllFactoriesValid` requires each alternative to satisfy exactly one concept, not the same one).

Consequence for kernel scope: the block-interleaved factory's kernels
(`reader_unary_pad_multicore_both_dims.cpp`, `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp`,
`data_movement/tilize/.../compute/tilize_wh.cpp`) are **not touched** by this port, and the
block-interleaved `c_1` self-loop disposition from the brief is **not** applied.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — each factory exposes
  `static ProgramDescriptor create_descriptor(const TilizeWithValPaddingParams&, const Tensor&, Tensor&)`.
- Variants: four factories on one `DeviceOperation` (`TilizeWithValPaddingDeviceOperation`),
  selected by `select_program_factory` (`device/tilize_with_val_padding_device_operation.cpp:43`):
  SingleCore, MultiCoreBlockInterleaved, MultiCoreDefault, MultiCoreSharded.
- Custom `compute_program_hash`: **none** — the op already uses the default reflection-based hash
  (audit confirmed; grep-clean). No device-op-class edit forced on that account.
- Custom `override_runtime_arguments` / `get_dynamic_runtime_args`: none.
- Pybind `create_descriptor`: none (`tilize_with_val_padding_nanobind.cpp` binds only the two
  top-level host functions), so no pybind surface is removed by this port.
- Op-owned tensors: none.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `MetalV2FactoryConcept`,
realized as `ProgramSpecFactoryConcept`. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

---

### Variant: SingleCore (`tilize_with_val_padding_single_core_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `…/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows.cpp` | `core_ranges` (single core: `sub_core_grids`'s first core, else `{0,0}`) | `[0]=tile_row_size_bytes`, `[1]=unpadded_row_size_bytes` (**never read by the kernel**), `[2…]=TensorAccessorArgs(*src0_buffer)` | none | 16 slots: `src0_buffer` (`Buffer*`), `input_w`, `padded_W_diff_blocks`, `input_z`, `padded_Z_diff_blocks`, `input_y`, `padded_Y_diff_blocks`, `num_leftover_Y`, `input_x`, `padded_row_size_bytes`, `packed_pad_value`, `num_blocks_w_input`, `num_blocks_w_output`, `num_blocks_w_diff`, `block_row_size`, `block_row_leftover_size` | none | none | `ReaderConfigDescriptor{}` |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `core_ranges` | `[0]=output_cb_index` (=`c_16`, a magic CB index), `[1…]=TensorAccessorArgs(*dst_buffer)` | none | 3 slots: `dst_buffer` (`Buffer*`), `num_tiles`, `0` (start_id) | none | none | `WriterConfigDescriptor{}` |
| compute | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | `core_ranges` | `[0]=num_tiles / num_tiles_per_block`, `[1]=num_tiles_per_block` | none | none | none | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_llk_acc, .unpack_to_dest_mode = v}` |

`v` = `vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, Default)` with `v[c_0] = UnpackToDestFp32`
iff `fp32_llk_acc` (`:188-191`). `fp32_llk_acc = in==FLOAT32 || in==FP8_E4M3 || out==FP8_E4M3 || out==BFLOAT8_B`.
Every other `ComputeConfigDescriptor` field is left at its legacy default.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`src0_cb_index = 0`) | `num_input_tiles * input_single_tile_size` | `core_ranges` | `input_cb_data_format` | `input_single_tile_size` | not set |
| `c_16` (`output_cb_index`) | `num_output_tiles * output_single_tile_size` | `core_ranges` | `output_cb_data_format` | `output_single_tile_size` | not set |

`num_input_tiles == num_output_tiles == num_tiles_per_block`.

#### Semaphores

none — the op uses no semaphores in any factory.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `…single_core_program_factory.cpp:138` (`TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args)`) | input (`a`) | reader RTA slot 0 (`src0_buffer`) |
| `…single_core_program_factory.cpp:173` (`TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)`) | output | writer RTA slot 0 (`dst_buffer`) |

Both are 2-arg kernel-side constructions (`TensorAccessor(args, addr)`) → **Case 1**; no 3rd (page-size)
argument anywhere in this op.

#### Work split

n/a — single core.

---

### Variant: MultiCoreDefault (`tilize_with_val_padding_multi_core_default_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `…/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp` | `all_cores` | `[0]=shift_bits`, `[1]=unpadded_row_size_bytes`, `[2]=elem_size`, `[3]=num_pages_in_row`, `[4]=page_size`, `[5]=aligned_page_size` (**never read by the kernel** — audit "Misc anomalies"), `[6]=size_of_valid_data_in_last_page_in_row`, `[7…]=TensorAccessorArgs(*src0_buffer)` | none | per core: `src0_buffer` (`Buffer*`), `padded_row_size_bytes`, `packed_pad_value`, `start_page_id`, `assignment.size()` (`n_block_reps`), then a **variable-length** block-representation stream of `5 × <distinct consecutive BlockRep groups>` values (`:193-229`) | none | none | `ReaderConfigDescriptor{}` |
| writer | `…/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | `[0]=output_cb_index` (=`c_16`), `[1…]=TensorAccessorArgs(*dst_buffer)` | none | per core: `dst_buffer` (`Buffer*`), `num_tiles_per_core`, `tile_start_id` | none | none | `WriterConfigDescriptor{}` |
| compute | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | `core_range` (only if non-empty) | `[0]=nblocks_per_core`, `[1]=num_tiles_per_row` | none | none | none | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_llk_acc, .unpack_to_dest_mode = v}` |
| compute_cliff | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | `core_range_cliff` (only if `has_cliff`) | `[0]=nblocks_per_core_cliff`, `[1]=num_tiles_per_row` | none | none | none | none | same `ComputeConfigDescriptor` |

`v` built identically to SingleCore (`:139-142`).

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `num_tiles_per_row * input_single_tile_size` | `all_cores` | `input_cb_data_format` | `input_single_tile_size` | not set |
| `c_16` | `num_tiles_per_row * output_single_tile_size` | `all_cores` | `output_cb_data_format` | `output_single_tile_size` | not set |

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `…multi_core_default_program_factory.cpp:113` | input (`a`) | reader RTA slot 0, per core |
| `…multi_core_default_program_factory.cpp:127` | output | writer RTA slot 0, per core |

Both **Case 1**, 2-arg.

#### Work split

- Driver: `ttnn::split_blocks_for_tilize(available_grid, num_blocks)` (`:46-47`).
- `num_cores` = `ncores`; `all_cores` = `core_range ∪ core_range_cliff` (verified in
  `work_split_tilize.hpp:334-364` — `all_cores` is literally seeded from `core_range` and then the
  cliff core is inserted).
- `core_group_1` = `core_range`, count_per_core = `nblocks_per_core`.
- `core_group_2` = `core_range_cliff` (0 or 1 core), count_per_core = `nblocks_per_core_cliff`.
- Per-core RTA values come from `ttnn::distribute_work(...)` (`:174-181`) walked over
  `corerange_to_cores(available_grid)[0 … ncores-1]`.

---

### Variant: MultiCoreSharded (`tilize_with_val_padding_multi_core_sharded_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `…/kernels/dataflow/reader_unary_pad_height_width_sharded.cpp` | `all_cores` (= output shard grid) | `[0]=src0_cb_index` (`c_1`), `[1]=src1_cb_index` (`c_0`), `[2]=src2_cb_index` (`c_2`) — three magic CB indices | none | per core (same values on every core): `num_input_rows`, `input_shard_width_bytes`, `(num_input_rows/num_batches)*input_shard_width_bytes`, `ntiles_per_batch`, `num_padded_rows`, `num_batches`, `packed_pad_value` | none | none | `ReaderConfigDescriptor{}` |
| writer | `…/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `all_cores` | `[0]=output_cb_index` (`c_16`) | none | per core: `ntiles_per_core` | none | none | `WriterConfigDescriptor{}` |
| compute | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | `all_cores` | `[0]=nblocks_per_core`, `[1]=ntiles_per_block` | none | none | none | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_llk_acc, .unpack_to_dest_mode = v}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | notes |
|---|---|---|---|---|---|---|
| `c_1` (`src0_cb_index`) | `num_input_rows * input_shard_width_bytes` | `all_cores` | `input_cb_data_format` | `input_shard_width_bytes` | not set | **borrowed memory**: `cb_src0.buffer = a.buffer()` when `src_sharded` (`:74-76`) |
| `c_0` (`src1_cb_index`) | `ntiles_per_batch * 2 * input_single_tile_size` | `all_cores` | `input_cb_data_format` | `input_single_tile_size` | not set | staging (tilize input) |
| `c_2` (`src2_cb_index`) | `input_shard_width_bytes` | `all_cores` | `input_cb_data_format` | `input_shard_width_bytes` | not set | pad scratch |
| `c_16` (`output_cb_index`) | `ntiles_per_core * output_single_tile_size` | `all_cores` | `output_cb_data_format` | `output_single_tile_size` | not set | **borrowed memory**: `cb_output.buffer = dst_buffer` when `out_sharded` (`:110-112`) |

No `CBDescriptor::address_offset`, no `GlobalCircularBuffer` anywhere (audit-confirmed; re-verified by
reading all four factories).

#### Tensor accessors

none — the sharded reader/writer/compute build no `TensorAccessor`. Tensor bases reach the device
purely through the two borrowed-memory CBs, so this factory passes **no address RTA at all**.

#### Work split

n/a — one kernel set over the output shard grid; every core gets identical RTA values.

---

### Shared kernels

| kernel source | relationship | `_metal2` fork beside it? | rung taken |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | *borrowed* — bound by ~24 other op directories (`copy/typecast`, `data_movement/{bcast,concat,copy,permute,reshape_on_device,slice,tilize,transpose}`, `embedding`, `kv_cache`, `matmul`, `reduction/{generic,prod}`, several `experimental/*`, …) | no (the only existing fork is under `experimental/quasar/`, which is off-limits) | **create the fork** beside the original |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | *borrowed* — bound by `sharded/interleaved_to_sharded`, `sharded_partial/interleaved_to_sharded_partial`, `data_movement/{tilize,transpose,untilize}`, `experimental/padded_slice`, `experimental/transformer/nlp_kv_cache_load_slice`, `reduction/generic` | no | **create the fork** beside the original |
| `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | *borrowed* from the shared kernel pool `ttnn/cpp/ttnn/kernel/` — also bound by all three `data_movement/tilize` factories | no | **create the fork** beside the original |
| `…/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_*.cpp` | *own*, not bound by any other op (`grep -rl` over `ttnn/cpp/ttnn/operations` returns only this op) | n/a | convert **in place** |

`data_movement/common/kernels/common.hpp` (`tt_memmove`) is called only from the block-interleaved
reader, which this pass does not port — so no kernel-lib escape is exercised.

Each fork gets a pointer comment in the legacy original per
*Caution: Porting a shared kernel*; each is reported under "Open items for downstream".

### Flags

- **Two positional CTAs are emitted but never read by their kernel**: single-core reader slot 1
  (`unpadded_row_size_bytes`, `…single_core_program_factory.cpp:137` vs kernel
  `reader_unary_pad_dims_split_rows.cpp:13-14`) and multicore reader slot 5 (`aligned_page_size`,
  `…multi_core_default_program_factory.cpp:105-112` vs kernel
  `reader_unary_pad_dims_split_rows_multicore.cpp:67-73`). In legacy they are load-bearing
  *positionally* (they shift the `TensorAccessorArgs<N>` offset). **Decision: carry them across as
  named CTAs, unchanged.** Deleting them would be a (provably behaviour-neutral) functional cleanup,
  which the audit explicitly routed to the ops team rather than the port; keeping them makes the diff
  a pure syntax swap. Recorded in the port report as a drop candidate.
- **Sharded RTAs are node-invariant.** All seven sharded reader RTAs and the single writer RTA take
  the same value on every core, so they are really CRTAs. Per the recipe's explicit instruction, the
  port does **not** convert RTA→CRTA (it changes dispatch semantics); noted for a later pass.
- The comment drift the audit flagged ("Assuming bfloat16 dataformat" at
  `…single_core_program_factory.cpp:59-60` and the multicore equivalent) is preserved verbatim.
- `reader_unary_pad_multicore_both_dims.cpp` is referenced only by the excluded block-interleaved
  factory, so it is not audited or converted here.
- **Unreferenced file already in the op directory (pre-existing, not touched):**
  `device/factories/tilize_with_val_padding_shared_variables.hpp` declares
  `struct shared_variables_interleaved` which nothing in the repo includes — leftover from the
  pre-`ProgramDescriptor` era. Left exactly as found; noted so the report makes clear it was not
  audited or converted.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (the brief's `MetalV2FactoryConcept`)
  — `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const TilizeWithValPaddingParams&, const Tensor&, Tensor&)`
  replaces `create_descriptor` on each ported factory. No `override_runtime_arguments`, so the base
  (not `Custom…`) spec concept applies and cache hits refresh only tensor bindings.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - `MultiCoreBlockInterleavedFactory` keeps `create_descriptor` and stays on
    `ProgramDescriptorFactoryConcept`; the `program_factory_t` variant is therefore mixed-concept.
    Each alternative still satisfies exactly one concept, so `AllFactoriesValid` holds.
  - The three ported factory headers drop `#include <tt-metalium/program_descriptors.hpp>` in favour
    of `ttnn/metal_v2_artifacts.hpp`; the block-interleaved header keeps it.

## Planned Spec Shape

### Variant: SingleCore

- **DataflowBufferSpecs** (2): `IN{"in"}` (`entry_size = input_single_tile_size`,
  `num_entries = num_input_tiles`, `data_format_metadata = input_cb_data_format`);
  `OUT{"out"}` (`entry_size = output_single_tile_size`, `num_entries = num_output_tiles`,
  `data_format_metadata = output_cb_data_format`). No tile metadata (legacy `.tile` unset).
- **KernelSpecs** (3): `READER{"reader"}`, `WRITER{"writer"}`, `COMPUTE{"compute"}` — 1:1 with the
  legacy `KernelDescriptor`s.
  - reader: `IN` PRODUCER (accessor `in0`); `TensorBinding{INPUT, "src"}`; 1 named CTA
    (`bytes_per_tile_row`) + the carried-over unread `unpadded_row_size_bytes`; 15 named RTAs.
  - writer: `OUT` CONSUMER (accessor `out`); `TensorBinding{OUTPUT, "dst"}`; 2 named RTAs
    (`num_pages`, `start_id`); no CTAs (the CB index became the binding, the accessor args are
    auto-injected).
  - compute: `IN` CONSUMER (accessor `in`), `OUT` PRODUCER (accessor `out`); 2 named CTAs.
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `INPUT{"input"}` = `a.tensor_spec()`, `OUTPUT{"output"}` = `output.tensor_spec()`.
- **WorkUnitSpecs** (1): `{READER, WRITER, COMPUTE}` on `core_ranges`.

### Variant: MultiCoreDefault

- **DataflowBufferSpecs** (2): `IN`/`OUT` as above with the multicore sizes
  (`num_entries = num_tiles_per_row` each).
- **KernelSpecs** (up to 4): `READER`, `WRITER`, `COMPUTE{"compute"}` (iff `core_range` non-empty),
  `COMPUTE_CLIFF{"compute_cliff"}` (iff `has_cliff`) — preserving the legacy per-group CTA split.
  - reader: `IN` PRODUCER (`in0`); `TensorBinding{INPUT, "src"}`; 6 named CTAs (+ the unread
    `aligned_page_size`); 4 named RTAs (`padded_X_size`, `pad_value`, `start_page_id`,
    `n_block_reps`) **plus a per-node runtime-vararg block** for the block-representation stream.
  - writer: `OUT` CONSUMER (`out`); `TensorBinding{OUTPUT, "dst"}`; 2 named RTAs.
  - compute / compute_cliff: `IN` CONSUMER (`in`), `OUT` PRODUCER (`out`); 2 named CTAs each.
- **TensorParameters** (2): `INPUT`, `OUTPUT`.
- **WorkUnitSpecs** (1 or 2): `{READER, WRITER, COMPUTE}` on `core_range` and/or
  `{READER, WRITER, COMPUTE_CLIFF}` on `core_range_cliff` — their union is exactly `all_cores`, so
  reader/writer placement is unchanged.

### Variant: MultiCoreSharded

- **DataflowBufferSpecs** (4):
  - `SRC_SHARD{"src_shard"}`: `entry_size = input_shard_width_bytes`, `num_entries = num_input_rows`,
    `data_format_metadata = input_cb_data_format`, `borrowed_from = INPUT` (iff `src_sharded`).
  - `STAGE{"stage"}`: `entry_size = input_single_tile_size`, `num_entries = ntiles_per_batch * 2`.
  - `PAD{"pad"}`: `entry_size = input_shard_width_bytes`, `num_entries = 1`.
  - `OUT_SHARD{"out_shard"}`: `entry_size = output_single_tile_size`, `num_entries = ntiles_per_core`,
    `data_format_metadata = output_cb_data_format`, `borrowed_from = OUTPUT` (iff `out_sharded`).
- **KernelSpecs** (3): `READER`, `WRITER`, `COMPUTE`.
  - reader: `SRC_SHARD` **self-loop** (PRODUCER `in0` + CONSUMER `in0`), `STAGE` PRODUCER (`in1`),
    `PAD` **self-loop** (PRODUCER `pad` + CONSUMER `pad`); 7 named RTAs; no CTAs.
  - writer: `OUT_SHARD` CONSUMER (`out`); 1 named RTA (`num_units`); no CTAs.
  - compute: `STAGE` CONSUMER (`in`), `OUT_SHARD` PRODUCER (`out`); 2 named CTAs.
- **TensorParameters** (2): `INPUT`, `OUTPUT` — declared *only* to back the two borrowed DFBs (no
  kernel `TensorBinding`; `program_spec.cpp:540-541` registers a `borrowed_from` parameter as used).
- **WorkUnitSpecs** (1): `{READER, WRITER, COMPUTE}` on `all_cores`.

### CB-endpoint census (re-derived from the kernel touches, not transcribed)

| variant | DFB | touchers (kernel : ops) | disposition | agrees with brief? |
|---|---|---|---|---|
| SingleCore | `IN` | reader: `reserve_back`/`push_back`/`get_write_ptr`; compute: `wait_front`/`pop_front` (inside `compute_kernel_lib::tilize`) | 1P + 1C | yes |
| SingleCore | `OUT` | compute: `reserve_back`/`push_back`; writer: `wait_front`/`pop_front` | 1P + 1C | yes |
| MultiCoreDefault | `IN` | reader P; compute **or** compute_cliff C (disjoint node sets) | 1P + 1C per node — two CONSUMER KernelSpecs over non-overlapping nodes, same kernel kind, identical binding params → legal without any flag | yes (brief did not call it multi-binding) |
| MultiCoreDefault | `OUT` | compute/compute_cliff P (disjoint); writer C | 1P + 1C per node | yes |
| MultiCoreSharded | `SRC_SHARD` | reader only (`reserve_back(num_input_rows)` at `:31`, `get_read_ptr()` at `:35`) | **self-loop** on reader (+ `borrowed_from` input) | yes |
| MultiCoreSharded | `STAGE` | reader P (`reserve_back`/`get_write_ptr`/`push_back`); compute C | 1P + 1C | yes |
| MultiCoreSharded | `PAD` | reader only (`reserve_back(1)` at `:33`, `get_write_ptr()` at `:37`) | **self-loop** on reader | yes |
| MultiCoreSharded | `OUT_SHARD` | compute P; writer C (`wait_front`/`pop_front`) | 1P + 1C (+ `borrowed_from` output) | yes |

No dead CB, and **no** `allow_instance_multi_binding` anywhere: every census fits 1P+1C or a
single-toucher self-loop.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| MultiCoreDefault `compute_desc` (`core_range`, CTA `nblocks_per_core`) + `compute_cliff_desc` (`core_range_cliff`, CTA `nblocks_per_core_cliff`), both `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | `COMPUTE`, `COMPUTE_CLIFF` (same forked source, distinct per-group named CTAs) | `wu_main` (`core_range`), `wu_cliff` (`core_range_cliff`) | `IN` (each binds CONSUMER), `OUT` (each binds PRODUCER) — disjoint node coverage, so one producer + one consumer per node instance; **not** a multi-binding, **not** a self-loop |

The per-group `nblocks_per_core` / `nblocks_per_core_cliff` stays a **CTA** on each KernelSpec (it is a
template argument to `compute_kernel_lib::tilize`'s caller loop bound and was a CTA in legacy);
demoting it to an RTA to collapse the two specs is the anti-pattern this row exists to prevent.

SingleCore and MultiCoreSharded have no work-split multiplicity.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `single_core…:138` / `multi_core_default…:113` | `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` | `TensorParameter{INPUT}` + `TensorBinding{INPUT, "src"}`; accessor CTAs auto-injected |
| `single_core…:173` / `multi_core_default…:127` | `TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args)` | `TensorParameter{OUTPUT}` + `TensorBinding{OUTPUT, "dst"}` |
| `single_core…:152-154` reader RTA slot 0 | `src0_buffer` (`Buffer*` → auto `BufferBinding`) | the `TensorBinding`'s implicit base-address CRTA |
| `single_core…:183` writer RTA slot 0 | `dst_buffer` | ditto |
| `multi_core_default…:195` reader RTA slot 0 (per core) | `src0_buffer` | ditto |
| `multi_core_default…:232` writer RTA slot 0 (per core) | `dst_buffer` | ditto |
| `reader_unary_pad_dims_split_rows.cpp:14,20,44` | `TensorAccessorArgs<2>()` + `get_arg_val<uint32_t>(0)` + `TensorAccessor(src_args, src_addr)` | `TensorAccessor(tensor::src)` |
| `reader_unary_pad_dims_split_rows_multicore.cpp:73,75,84` | `TensorAccessorArgs<7>()` + `src_addr` RTA + 2-arg accessor | `TensorAccessor(tensor::src)` |
| `writer_unary_interleaved_start_id.cpp:11,16,31` | `TensorAccessorArgs<1>()` + `dst_addr` RTA + 2-arg accessor | `TensorAccessor(tensor::dst)` (in the `_metal2` fork) |
| `single_core…:172` / `multi_core_default…:126` writer CTA slot 0 | `output_cb_index` (magic CB index `c_16`) | `DFBBinding{OUT, "out", CONSUMER}` |
| `reader_unary_pad_dims_split_rows.cpp:17` | `constexpr uint32_t dfb_id_in0 = 0;` (hardcoded CB index) | `DataflowBuffer dfb_in0(dfb::in0)` |
| `reader_unary_pad_dims_split_rows_multicore.cpp:64` | `constexpr uint32_t dfb_id_in0 = 0;` | `DataflowBuffer dfb_in0(dfb::in0)` |
| `writer_unary_interleaved_start_id.cpp:15` | `cb_id_out = get_compile_time_arg_val(0)` | `dfb::out` |
| `writer_unary_sharded.cpp:12` | `dfb_id_out = get_compile_time_arg_val(0)` | `dfb::out` |
| `sharded…:118-122` reader CTAs 0/1/2 | `src0_cb_index`, `src1_cb_index`, `src2_cb_index` (three magic CB indices) | `DFBBinding`s for `SRC_SHARD`/`STAGE`/`PAD` |
| `sharded…:135` writer CTA 0 | `output_cb_index` | `DFBBinding{OUT_SHARD, "out", CONSUMER}` |
| `sharded…:75` / `:111` | `cb_src0.buffer = a.buffer()` / `cb_output.buffer = dst_buffer` | `DataflowBufferSpec::borrowed_from = INPUT / OUTPUT` |
| `ttnn/cpp/ttnn/kernel/compute/tilize.cpp:16,19,23-30` | `tt::CBIndex::c_0` / `c_16` hardcoded in `compute_kernel_hw_startup`, `is_fp32_input_format<>`, `tilize<>` | `dfb::in` / `dfb::out` (implicit `constexpr operator uint32_t` bridges to the kernel-lib's `uint32_t` template params) |
| `writer_unary_interleaved_start_id.cpp:19` | `get_local_cb_interface(cb_id_out).fifo_page_size` | `dfb.get_entry_size()` (identical on the DM path: `cb_addr_shift == 0` outside TRISC, `dataflow_buffer.inl:35-40`) |
| all reader/writer positional RTAs | `get_arg_val<uint32_t>(N)` | `get_arg(args::<name>)` |
| all reader/compute positional CTAs | `get_compile_time_arg_val(N)` | `get_arg(args::<name>)` |
| `multi_core_default…:193-229` reader RTA tail | positional block-representation stream read at a running `rt_arg_idx` | `KernelAdvancedOptions::num_runtime_varargs_per_node` + `AdvancedKernelRunArgs::runtime_varargs`, read with `get_vararg(i)` |

**Page-size 3rd-argument CTAs/RTAs**: none — every accessor in this op is 2-arg (audit-confirmed).
**Semaphore-ID RTAs**: none — the op has no semaphores.

## Applied Patterns

- **Sync-free / single-ended CB → self-loop DFB**: sharded `SRC_SHARD` (borrowed input; the reader
  only reserves and peeks its read pointer) and sharded `PAD` (reader-private scratch) are each bound
  PRODUCER *and* CONSUMER on the sharded reader. Legal on Gen1 for DM kernels.
- **Borrowed-memory DFB**: sharded `SRC_SHARD` / `OUT_SHARD` take `borrowed_from` naming the input /
  output `TensorParameter`, replacing the legacy `CBDescriptor::buffer` assignment. No
  `dfb_run_overrides` entry is needed — the backing L1 address resolves from the `TensorArgument`.
- **Preserved work-split multiplicity**: two same-source compute KernelSpecs in MultiCoreDefault over
  disjoint node sets, each keeping its own per-group CTA.
- **Porting a shared kernel** (three instances, all on the *create the fork* rung): the two donor
  writers and the pooled tilize compute kernel are forked `_metal2` beside their originals, with a
  pointer comment added to each original. The `experimental/quasar/` copies of two of these are
  off-limits and were not consulted.
- **Runtime varargs for a genuine indexed collection**: the MultiCoreDefault reader's
  block-representation stream is a variable-count, loop-indexed run whose length is a runtime value
  (`n_block_reps`), so it stays a vararg block rather than being named. Because the length differs
  per core, the schema uses the (deprecated but purpose-built) per-node vararg-count override rather
  than a single scalar count.
- **Compute `hw_config` Style B** (the op sets a Metal `ComputeConfigDescriptor` directly, with no
  TTNN `ComputeKernelConfig` in play): build `ComputeGen1Config` by hand,
  `enable_32_bit_dest = fp32_llk_acc`, and translate the legacy `unpack_to_dest_mode` vector's single
  non-default entry (`v[c_0] = UnpackToDestFp32`) to `unpack_modes = {{<in-DFB>, UnpackToDest}}`.
  Every other field is left at its `ComputeGen1Config` default, which matches the legacy default.
  Do **not** route through `to_compute_hardware_config` (its defaults are the high-performance ones).
- **DM `hw_config` defaults**: every reader is `ReaderConfigDescriptor{}` and every writer
  `WriterConfigDescriptor{}` — the resolved reader/writer default triples — so both take the
  arch-agnostic TTNN helpers `create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)`.

## Deferred / Flagged

- **New finding — `unpack_modes` legality is safe here by construction, but only just.** The Metal 2.0
  validator rejects `UnpackToDest` on a consumed DFB whose format is ≤16-bit when
  `enable_32_bit_dest == false` (`program_spec.cpp:1032-1039`). In this op both the entry and
  `enable_32_bit_dest` are driven by the *same* `fp32_llk_acc` flag, so whenever the entry exists the
  32-bit-dest escape at `program_spec.cpp:1011-1012` applies and the entry is accepted — including
  the `bfloat16`-in / `bfloat8_b`-out case, where legacy also set `UnpackToDestFp32` on a 16-bit CB.
  Faithful, no capitulation needed; recorded so a future refactor of `fp32_llk_acc` knows the coupling
  is load-bearing.
- **New finding — per-node vararg counts.** The only mechanism for a per-node-varying vararg count is
  `KernelAdvancedOptions::num_runtime_varargs_per_node`, which carries a `[[deprecated]]` attribute
  ("will be removed once existing uses are refactored"). This op *is* such an existing use. Flagged
  in the port report as a handoff/friction item.
- Nothing else surfaced that the audit missed. No `GlobalCircularBuffer`, no `GlobalSemaphore`, no
  CTA varargs, no `address_offset`, no offset-folded base pointer, no Case 2 binding, no
  `TensorParameter` relaxation.
