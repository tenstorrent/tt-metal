# Port Plan — `ttnn/cpp/ttnn/operations/reduction/sampling`

Port plan for `sampling`, ported from the legacy `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `SamplingProgramFactory::create_descriptor` returns a plain
  `tt::tt_metal::ProgramDescriptor` (`device/sampling_program_factory.hpp:14-15`).
- Variants: single. `program_factory_t = std::variant<SamplingProgramFactory>`
  (`device/sampling_device_operation.hpp:24`).
- Custom `compute_program_hash`: **none** — already the default reflection-based hash. Confirmed by grep over the
  op directory; matches the audit.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `ProgramSpecFactoryConcept`. Carried
forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels

Three `KernelDescriptor`s per running core-set, all sources owned by this op. `args_base` below is the CTA index
after the four `TensorAccessorArgs` blocks the writer prepends.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (×1) | `device/kernels/dataflow/reader_values_indices_tensor.cpp` | `core_grid` (all `num_cores` active cores), created once | 0 `input_values_cb_index`(c_0) · 1 `final_indices_rm_cb_index`(c_12) · 2 `index_cb_index`(c_2) · 3 `Ht` · 4 `Wt` · 5 `aligned_final_indices_rm_unit_size` · 6 `tile_height` · 7 `use_32bit_index` · 8 `num_users` · then `TensorAccessorArgs(input_values)`, `TensorAccessorArgs(input_indices)` (`:352-363`) | none | per core, 2 slots: `{input_values_tensor, input_indices_tensor}` — whole `MeshTensor`s via `emplace_runtime_args` (`:374-379`) | none | none | absent → resolves **O2** (DM) | `ReaderConfigDescriptor{}` → resolved `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` = reader default |
| writer (×`num_cores`, one per core) | `device/kernels/dataflow/writer_interleaved.cpp` | `CoreRangeSet single_core{CoreRange(core, core)}` (`:385`, `:419`) | `TensorAccessorArgs(output)`, `(temp)`, `(k)`, `(p)` first (`:388-391`), then `args_base+`: 0 `output_cb_index`(c_13) · 1 `topk_mask_cb_index`(c_4) · 2 `scaler_max_cb_index`(c_3) · 3 `scaler_sum_cb_index`(c_17) · 4 `final_indices_rm_cb_index`(c_12) · 5 `cb_local_vals_index`(c_1) · 6 `output_ind_cb_index`(c_8) · 7 `aligned_final_indices_rm_unit_size` · 8 `aligned_out0_unit_size` (**never read by the kernel**) · 9 `rand_tile_index`(c_11) · 10 `k_cb_index`(c_14) · 11 `p_cb_index`(c_15) · 12 `temp_cb_index`(c_16) · 13 `i` (→ `core_id`) · 14 `tile_width` (→ `ids_per_batch`) · 15 `num_cores` · 16 `use_32bit_index` · 17 `num_users` (`:392-413`) | none | 4 slots: `{output_mesh, temp, k, p}` — whole `MeshTensor`s (`:422`) | none | none | absent → resolves **O2** (DM) | `WriterConfigDescriptor{}` → resolved `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = writer default |
| compute (×`num_cores`, one per core) | `device/kernels/compute/sampling.cpp` | `single_core` (`:450`) | 0 `input_values_cb_index`(c_0) · 1 `index_cb_index`(c_2) · 2 `input_transposed_cb_index`(c_5) · 3 `index_transposed_cb_index`(c_6) · 4 `values_cb_index`(c_7) · 5 `output_ind_cb_index`(c_8) · 6 `topk_mask_cb_index`(c_4) · 7 `scaler_max_cb_index`(c_3) · 8 `scaler_sum_cb_index`(c_17) · 9 `cb_cur_max_index`(c_9) · 10 `cb_cur_sum_index`(c_10) · 11 `Ht` · 12 `Wt` · 13 `logWt` · 14 `rand_tile_index`(c_11) · 15 `random_seed` · 16 `cb_local_vals_index`(c_1) · 17 `temp_cb_index`(c_16) · 18 `tile_width` · 19 `stable_sort` (`:425-445`) | none | none | none | none | absent → resolves **O3** (compute) | `ComputeConfigDescriptor{.fp32_dest_acc_en = use_32bit_index}` (`:454-456`); every other field left at its legacy default |

**All 21 per-core compute `KernelDescriptor`s are byte-identical apart from `core_ranges`** — `compute_args` contains
no per-core value. Only the writer carries a per-core CTA (`i` → `core_id`).

### CBs

18 `CBDescriptor`s, every one over `core_grid`, every one with a single-element `format_descriptors` (so no aliasing)
and no `tile` field set (so `tile_format_metadata` stays `nullopt`). No `CBDescriptor` sets `global_circular_buffer`,
`address_offset`, `buffer` or `tensor` — all keep their struct defaults.

| index | name (host var) | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|---|
| c_0 | `input_values` | `cb_in_units(4) * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` (Float16_b) | `input_values_tile_size` | — |
| c_1 | `cb_local_vals` | `num_cb_unit(2) * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` | `input_values_tile_size` | — |
| c_2 | `index` | `cb_in_units(4) * index_tile_size` | `core_grid` | `index_cb_data_format` (UInt16 on WH/BH, Int32 otherwise) | `index_tile_size` | — |
| c_3 | `scaler_max` | `scale_tiles(1) * scalar_tile_size` | `core_grid` | `scalar_df` (Float16_b; Float32 only if input were FLOAT32) | `scalar_tile_size` | — |
| c_4 | `topk_mask` | `cb_in_units(4) * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` | `input_values_tile_size` | — |
| c_5 | `input_transposed` | `Wt * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` | `input_values_tile_size` | — |
| c_6 | `index_transposed` | `Wt * index_tile_size` | `core_grid` | `index_cb_data_format` | `index_tile_size` | — |
| c_7 | `values` | `num_cb_unit(2) * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` | `input_values_tile_size` | — |
| c_8 | `output_ind` | `num_cb_unit(2) * index_tile_size` | `core_grid` | `index_cb_data_format` | `index_tile_size` | — |
| c_9 | `cb_cur_max` | `num_out_tiles(Ht) * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` | `input_values_tile_size` | — |
| c_10 | `cb_cur_sum` | `num_out_tiles(Ht) * input_values_tile_size` | `core_grid` | `input_values_cb_data_format` | `input_values_tile_size` | — |
| c_11 | `rand_tile` | `rand_tile_size` | `core_grid` | `Float16_b` | `rand_tile_size` | — |
| c_12 | `final_indices_rm` | `Ht * tile_height * aligned_final_indices_rm_unit_size` | `core_grid` | `input_indices_cb_data_format` | `aligned_final_indices_rm_unit_size` | — |
| c_13 | `output` | `aligned_out0_unit_size` | `core_grid` | `index_cb_data_format` | `aligned_out0_unit_size` | — |
| c_14 | `k` | `k_chunk_size` (`num_cores * 4`) | `core_grid` | `k_cb_data_format` | `k_chunk_size` | — |
| c_15 | `p` | `p_chunk_size` (`num_cores * 2`) | `core_grid` | `p_cb_data_format` | `p_chunk_size` | — |
| c_16 | `temp` | `temp_chunk_size` (`num_cores * 2`) | `core_grid` | `temp_cb_data_format` | `temp_chunk_size` | — |
| c_17 | `scaler_sum` | `scale_tiles(1) * scalar_tile_size` | `core_grid` | `scalar_df` | `scalar_tile_size` | — |

### Semaphores

none — `desc.semaphores` is never populated.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/sampling_program_factory.cpp:362` | `input_values` (input) | reader slot 0 (`:377`) |
| `device/sampling_program_factory.cpp:363` | `input_indices` (input) | reader slot 1 (`:378`) |
| `device/sampling_program_factory.cpp:388` | `output` (output) | writer slot 0 (`:422`) |
| `device/sampling_program_factory.cpp:389` | `temp` (input) | writer slot 1 (`:422`) |
| `device/sampling_program_factory.cpp:390` | `k` (input) | writer slot 2 (`:422`) |
| `device/sampling_program_factory.cpp:391` | `p` (input) | writer slot 3 (`:422`) |

Device-side accessor construction sites: reader `:74` (`s0`), `:76` (`s1`); writer `:93` (`addrg_k`), `:104`
(`addrg_p`), `:115` (`addrg_temp`), `:239` (`s_out`). All six use the 2-argument `(args, addr)` form.

### Work split

- Driver: **no `split_work_to_cores`.** One core per user:
  `num_cores = num_users = input_shape[2]` (`:77-80`), `core_grid = num_cores_to_corerangeset(...)` or
  `sub_core_grids` narrowed to exactly the active cores (`:83-100`), `cores = corerange_to_cores(core_grid,
  num_cores, true)` (`:88`).
- num_cores: `num_users`, in `[1, 32]` (validation, `device/sampling_device_operation.cpp:75-78`).
- core_group_1 / core_group_2: n/a — a single homogeneous group. Every core does identical work; the only
  per-core value is the core's own index, delivered as the writer's `core_id` CTA.

### Shared kernels

none. All three `kernel_source` paths are inside this op's own directory (`:366-367`, `:416-417`, `:448`).
`grep -rl` on each filename across `ttnn/cpp/ttnn/operations/` finds only this factory, and no `*_metal2.cpp`
sibling exists beside any of them. So: no fork to reuse, none to create, no sunset list.

### Flags

- **Dead writer CTA.** `aligned_out0_unit_size` is emitted as writer CTA `args_base + 8` (`:403`) and documented as
  unused rather than read (`writer_interleaved.cpp:59`). Kept in the port as a named CTA (`out_stick_size`) — this is
  the audit's *Misc anomalies* item 2, which routes to the ops team, not into this diff.
- **Dead writer local.** `uint32_t arg_id = 0;` (`writer_interleaved.cpp:44`) is assigned and never read. It exists
  only to walk a positional RTA list that the port deletes outright, so it disappears with the RTAs rather than as a
  cleanup of its own.
- **Commented-out FIFO calls** around the `temp` staging read (`writer_interleaved.cpp:116`, `:121`). Left exactly as
  they are per the brief.
- No unreferenced kernel files in the op directory.
- `std::log2(Wt)` computes `logWt` through a floating-point path (`:439`). Carried through unchanged; audit *Misc
  anomalies* item 6.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — `create_program_artifacts` returning
  `ttnn::device_operation::ProgramArtifacts`.
- **Custom `compute_program_hash`**: none. Nothing to delete.
- **Implementation notes**:
  - `create_descriptor` is a `static` on `SamplingProgramFactory` and is **not** pybound
    (`sampling_nanobind.cpp` binds only the user-facing `ttnn::sampling`), so no pybind line has to be removed.
    The header's signature changes in place and `<tt-metalium/program_descriptors.hpp>` drops from both the factory
    header and the device-op header.
  - No op-owned tensors: the factory allocates nothing beyond the framework-allocated output, so
    `ProgramArtifacts::op_owned_tensors` stays defaulted.
  - **`ProgramRunArgs::kernel_run_args` is empty.** All six legacy RTAs are tensor addresses, so every one becomes a
    `TensorBinding`; no kernel is left with a runtime or common runtime argument. That is legal — a `KernelRunArgs`
    entry is required only for kernels that have runtime args
    (`tt_metal/api/tt-metalium/experimental/metal2_host_api/program_run_args.hpp:89-91`). Only `tensor_args` is
    populated.

## Planned Spec Shape

- **KernelSpecs (2 + `num_cores`)**:
  - `reader` — one spec, node set = all active cores.
  - `compute` — **one** spec, node set = all active cores. The legacy per-core compute descriptors carry no per-core
    CTA, so they collapse to one spec whose derived node set is exactly their union. This is not a CTA demotion:
    there is no per-group dimension to demote, and no argument moves from compile time to runtime.
  - `writer_<i>` — one spec per running core, `i ∈ [0, num_cores)`, differing only in the `core_id` CTA. Preserved
    per the brief: `core_id` stays a compile-time arg on a per-node spec.
- **DataflowBufferSpecs (18)**: one per legacy `CBDescriptor`, `entry_size = page_size` and
  `num_entries = total_size / page_size` (every legacy CB's `total_size` is an exact multiple of its `page_size`).
  `data_format_metadata` copied from the legacy `data_format`; `tile_format_metadata` left unset because no legacy
  `CBFormatDescriptor` sets `tile`. No aliasing, no `borrowed_from`, no `allow_instance_multi_binding`.
- **SemaphoreSpecs**: none — the legacy factory declares no semaphores.
- **TensorParameters (6)**: `input_values`, `input_indices`, `output`, `temp`, `k`, `p`. Strict `TensorSpec` matching
  (no relaxations); the audit recorded `TensorParameter relaxation = none`.
- **WorkUnitSpecs (`num_cores`)**: one per running core, `{reader, compute, writer_<i>}` targeting core `i`. Work
  units may not overlap in target nodes, and the writer specs are per-node, so per-node work units are the shape that
  fits. `reader` and `compute` are members of every work unit, which makes their derived node sets the union of all
  cores — matching the legacy `core_grid`. Per node: 2 DM kernels + 1 compute kernel, inside the Gen1 limits.
- **Op-owned tensors**: none.

### DFB endpoint assignment (re-derived from the kernel-touch census)

Per node exactly one reader, one writer and one compute instance are co-resident, and there is a single config path
(no sharding or layout branch), so each DFB carries one disposition. Census re-derived from the kernel sources; it
agrees with the brief on all 18 (10 legal 1:1, 8 self-loop, 0 multi-binding, 0 dead).

| DFB | PRODUCER (accessor) | CONSUMER (accessor) | disposition |
|---|---|---|---|
| `input_values` (c_0) | reader (`input_values`) | compute (`input_values`) | 1:1 |
| `index` (c_2) | reader (`index`) | compute (`index`) | 1:1 |
| `final_indices_rm` (c_12) | reader (`input_indices`) | writer_i (`final_indices`) | 1:1 |
| `scaler_max` (c_3) | writer_i (`scaler_max`) | compute (`scaler_max`) | 1:1 |
| `scaler_sum` (c_17) | writer_i (`scaler_sum`) | compute (`scaler_sum`) | 1:1 |
| `topk_mask` (c_4) | writer_i (`mask`) | compute (`topk_mask`) | 1:1 |
| `temp` (c_16) | writer_i (`temp`) | compute (`temp`) | 1:1 |
| `cb_local_vals` (c_1) | compute (`local_vals`) | writer_i (`local_values`) | 1:1 |
| `output_ind` (c_8) | compute (`output_ind`) | writer_i (`local_indices`) | 1:1 |
| `rand_tile` (c_11) | compute (`rand_tile`) | writer_i (`rand`) | 1:1 |
| `input_transposed` (c_5) | compute (`input_transposed`) | compute (`input_transposed`) | self-loop |
| `index_transposed` (c_6) | compute (`index_transposed`) | compute (`index_transposed`) | self-loop |
| `values` (c_7) | compute (`values`) | compute (`values`) | self-loop |
| `cb_cur_max` (c_9) | compute (`cur_max`) | compute (`cur_max`) | self-loop |
| `cb_cur_sum` (c_10) | compute (`cur_sum`) | compute (`cur_sum`) | self-loop |
| `output` (c_13) | writer_i (`out`) | writer_i (`out`) | self-loop |
| `k` (c_14) | writer_i (`k`) | writer_i (`k`) | self-loop |
| `p` (c_15) | writer_i (`p`) | writer_i (`p`) | self-loop |

Census notes worth recording, because each looks like an extra endpoint and is not:

- `temp` (c_16) is touched twice by the writer — a raw NoC read into its write pointer, then
  `generate_bcast_unary_scalar`'s own `reserve_back` / `push_back`. Both touches belong to the same kernel, so this is
  one producer, not two.
- `output` (c_13) has no FIFO operation at all: the writer only peeks a pointer and uses the buffer as a NoC write
  source. Single toucher → self-loop, with cosmetic roles.
- `k` (c_14) and `p` (c_15) are locked producers with no consumer: NoC read staging that the writer then reads back
  through a raw pointer. Single toucher each → self-loop.
- The three writer self-loops are legal on Gen1 (a DFB lowers to a plain circular buffer that one DM RISC both fills
  and drains) and only rejected on Gen2. The `num_cores` writer specs each self-loop their own single node, which the
  validator permits because the producer and consumer KernelSpec *sets* are identical.

Accessor names differ per kernel where the two kernels' own vocabularies differ (`final_indices_rm` is
`input_indices` to the reader and `final_indices` to the writer). That is the intended per-kernel-alias behaviour and
it keeps the kernel diffs minimal.

## Preserved Multiplicity

```
Legacy KernelDescriptors [writer × num_cores] of source device/kernels/dataflow/writer_interleaved.cpp
  → KernelSpecs [writer_0 … writer_{num_cores-1}] of same source, differing only in the core_id CTA
  → in WorkUnitSpecs [wu_0 … wu_{num_cores-1}], one single-node work unit each
  → sharing DFBs (endpoint role each KernelSpec binds):
      final_indices_rm CONSUMER · cb_local_vals CONSUMER · output_ind CONSUMER · rand_tile CONSUMER
      topk_mask PRODUCER · scaler_max PRODUCER · scaler_sum PRODUCER · temp PRODUCER
      output PRODUCER+CONSUMER · k PRODUCER+CONSUMER · p PRODUCER+CONSUMER
```

The writer specs cover **disjoint single-node** sets, so each node sees exactly one instance and each shared DFB is
an ordinary single-role binding per node. This is the disjoint-node work split, **not** the same-grid two-toucher
shape, and it needs no `allow_instance_multi_binding` flag.

The legacy per-core **compute** descriptors are not a multiplicity case: they carry identical CTAs, so they collapse
to one `KernelSpec` over the union of their nodes with no argument changing dispatch class.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `sampling_program_factory.cpp:362` | `TensorAccessorArgs(input_values_tensor).append_to(reader_compile_time_args)` | `TensorParameter{input_values}` + reader `TensorBinding` |
| `sampling_program_factory.cpp:363` | `TensorAccessorArgs(input_indices_tensor).append_to(...)` | `TensorParameter{input_indices}` + reader `TensorBinding` |
| `sampling_program_factory.cpp:388-391` | `TensorAccessorArgs(output_mesh/temp/k/p).append_to(writer_compile_time_args)` | `TensorParameter{output,temp,k,p}` + writer `TensorBinding`s |
| `sampling_program_factory.cpp:377` (reader RTA 0) | `MeshTensor` `input_values_tensor` pushed via `emplace_runtime_args` | `TensorBinding{input_values, "input_values"}` |
| `sampling_program_factory.cpp:378` (reader RTA 1) | `MeshTensor` `input_indices_tensor` | `TensorBinding{input_indices, "input_indices"}` |
| `sampling_program_factory.cpp:422` (writer RTA 0-3) | `MeshTensor`s `output_mesh, temp, k, p` | `TensorBinding`s `output`, `temp`, `k`, `p` |
| `reader_values_indices_tensor.cpp:52-53` | `get_arg_val<uint32_t>(0)`, `(1)` → `values_addr`, `indices_addr` | gone; `TensorAccessor(tensor::input_values)`, `TensorAccessor(tensor::input_indices)` |
| `reader_values_indices_tensor.cpp:68-69` | `TensorAccessorArgs<9>()` + `next_compile_time_args_offset()` chain | gone (binding carries the layout metadata) |
| `writer_interleaved.cpp:39-42` | `get_arg_val<uint32_t>(0..3)` → `dst_addr`, `temp_addr`, `k_addr`, `p_addr` | gone; `TensorAccessor(tensor::output/temp/k/p)` |
| `writer_interleaved.cpp:45-50` | `TensorAccessorArgs<0>()` + three `next_compile_time_args_offset()` links + `args_base` | gone |
| `writer_interleaved.cpp:44` | `uint32_t arg_id = 0;` (dead positional-RTA cursor) | gone with the positional RTAs it was for |
| reader CTA slots 0, 1, 2 (`:353-355`) | CB indices `c_0`, `c_12`, `c_2` | `DFBBinding`s → `dfb::input_values`, `dfb::input_indices`, `dfb::index` |
| writer CTA slots `args_base+0..6`, `+9..12` (`:395-407`) | CB indices `c_13, c_4, c_3, c_17, c_12, c_1, c_8, c_11, c_14, c_15, c_16` | `DFBBinding`s → `dfb::out`, `dfb::mask`, `dfb::scaler_max`, `dfb::scaler_sum`, `dfb::final_indices`, `dfb::local_values`, `dfb::local_indices`, `dfb::rand`, `dfb::k`, `dfb::p`, `dfb::temp` |
| compute CTA slots 0-10, 14, 16, 17 (`:426-442`) | CB indices `c_0, c_2, c_5, c_6, c_7, c_8, c_4, c_3, c_17, c_9, c_10, c_11, c_1, c_16` | `DFBBinding`s → `dfb::input_values`, `dfb::index`, `dfb::input_transposed`, `dfb::index_transposed`, `dfb::values`, `dfb::output_ind`, `dfb::topk_mask`, `dfb::scaler_max`, `dfb::scaler_sum`, `dfb::cur_max`, `dfb::cur_sum`, `dfb::rand_tile`, `dfb::local_vals`, `dfb::temp` |
| reader CTA slots 3-8 | positional `Ht, Wt, aligned_final_indices_rm_unit_size, tile_height, use_32bit_index, num_users` | named CTAs `Ht`, `Wt`, `input_indices_page_size`, `tile_height`, `use_32bit_index`, `num_users` |
| writer CTA slots `args_base+7, +8, +13..17` | positional `aligned_final_indices_rm_unit_size, aligned_out0_unit_size, i, tile_width, num_cores, use_32bit_index, num_users` | named CTAs `final_indices_stick_size`, `out_stick_size`, `core_id`, `ids_per_batch`, `num_cores`, `use_32bit_index`, `num_users` |
| compute CTA slots 11-13, 15, 18, 19 | positional `Ht, Wt, logWt, random_seed, tile_width, stable_sort` | named CTAs `Ht`, `Wt`, `logWt`, `seed`, `tile_width`, `stable_sort` |
| `writer_interleaved.cpp:242` | `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out)` | bare `DataflowBuffer` as the NoC source; the pointer-selection wrapper has no DFB twin and is unnecessary |

Named-CTA names are taken from the local variable each value was assigned to in the kernel, which is why a few differ
from the host's variable name (`tile_width` → `ids_per_batch` in the writer, `aligned_final_indices_rm_unit_size` →
`input_indices_page_size` in the reader and `final_indices_stick_size` in the writer).

**Semaphore-ID RTAs:** none — the op has no semaphores.
**Page-size third-argument CTAs/RTAs:** none — all six accessors already use the 2-argument form.
**Case 2 (raw pointer) bindings:** none — all six are Case 1, so no `get_bank_base_address` bridge is needed.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — the five compute-only intermediates (`input_transposed`, `index_transposed`, `values`, `cur_max`, `cur_sum`) run a
  genuine produce-then-consume cycle inside the compute kernel; each is bound PRODUCER **and** CONSUMER under one
  accessor name.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — `output` (sync-free address source, no FIFO ops at all), `k` and `p` (locked producer with no consumer). Each has
  exactly one toucher, the writer, so each self-loops. A DM self-loop is a Gen1-legal shape.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — `dfb::name` flows straight into `pack_tile`, `transpose_tile`, `reconfig_data_format*`, `add_tiles`,
  `mul_tiles_bcast_*`, `copy_tile`, `compute_kernel_lib::reduce<>`,
  `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<>` and `generate_mask<>`, in both call-argument and
  non-type-template-parameter position.

Not applied, and why: **Aliased DFBs** (no legacy CB carries more than one `buffer_index`); **Same-FIFO aliasing** (no
kernel reaches one buffer under two names); **Conditional / optional DFB bindings** (every DFB is bound
unconditionally — `use_32bit_index` and `stable_sort` change formats and LLK behaviour, never which DFBs exist);
**Multi-variant factories** (one factory, one code path); **Two-toucher 1P+1C** (no DFB has two distinct co-resident
touchers); **Demoting per-group CTA to RTA** (explicitly avoided — `core_id` stays a CTA on per-node writer specs);
**Porting a shared kernel** (no kernel is shared); **Removing pybound legacy factory entry points** (`create_descriptor`
was never pybound).

## Hardware configuration and compiler options

| kernel | legacy resolved config | Metal 2.0 |
|---|---|---|
| reader | `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` = reader default | `ttnn::create_reader_datamovement_config(device->arch())` |
| writer | `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = writer default | `ttnn::create_writer_datamovement_config(device->arch())` |
| compute | `ComputeConfigDescriptor{.fp32_dest_acc_en = use_32bit_index}`; `math_fidelity = HiFi4`, `math_approx_mode = false`, `dst_full_sync_en = false`, `bfp8_pack_precise = false`, `unpack_to_dest_mode = {}` all default | `ComputeGen1Config{.enable_32_bit_dest = use_32bit_index}` |

The compute config is built directly (Style B — the op sets a Metal `ComputeConfigDescriptor`, it does not resolve a
TTNN `ComputeKernelConfig`), so the remaining fields stay at `ComputeGen1Config`'s defaults, which coincide with the
legacy `ComputeConfigDescriptor` defaults field for field: `fpu_math_fidelity = HiFi4`,
`sfpu_precision_mode = Precise` (legacy `math_approx_mode = false`), `bfp_pack_precision_mode = Approximate` (legacy
`bfp8_pack_precise = false`), `double_buffer_dest = true` (legacy `dst_full_sync_en = false`, inverted), and an empty
`unpack_modes`.

`unpack_modes` needs no entry. The validator's required-entry rule fires only for a Float32 DFB consumed with
`enable_32_bit_dest = true`; validation pins the input values to BFLOAT16, so no DFB in this op carries Float32, and
on WH/BH `use_32bit_index` is false anyway. Legacy `unpack_to_dest_mode` was empty, which maps to omitting every entry.

`opt_level`: the reader and writer resolved to `O2`, which is `CompilerOptions`' default, so neither states it. The
compute kernel resolved to `O3` (the legacy `ComputeConfigDescriptor` default), which Metal 2.0 does **not** default
to, so the compute spec sets `KernelBuildOptLevel::O3` explicitly.

## Deferred / Flagged

- **New findings during planning:**
  - `generate_mask` (a cross-family donor) takes its fill base from `cb_mask.get_read_ptr()` after
    `reserve_back`, while the writer binds `topk_mask` as PRODUCER only. On Gen1 this is safe by construction:
    BRISC and NCRISC always initialise **both** FIFO pointers for every circular buffer in their slot mask
    (`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:503`, `ncrisc.cc:157` instantiate
    `setup_local_cb_read_write_interfaces<true, true, …>`), and the DFB's slot mask is
    `producer_risc_mask | consumer_risc_mask`, so a PRODUCER-only binding still gets a valid read pointer. Verified
    rather than assumed, because a binding-derived read/write split would have made this a latent wrong-address bug.
  - The same reasoning covers dropping `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out)`: a bare
    `DataflowBuffer` used as a NoC **source** resolves to `get_read_ptr()`
    (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:383-388`), where the legacy wrapper forced the write pointer.
    `output` (c_13) never runs a FIFO operation, so its read and write pointers both sit at the buffer base and the
    two spellings address the same bytes.
  - Building only the Gen1 compute config narrows this op's compute hardware config to WH/BH, even though its host and
    kernel code carry an architecture-gated 32-bit-index path aimed at Quasar. That is the recipe's Gen2-out-of-scope
    rule, and there is no generation-agnostic helper for a Style-B compute config; recorded for the port report.
