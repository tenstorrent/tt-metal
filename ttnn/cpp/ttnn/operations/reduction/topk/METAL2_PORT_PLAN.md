# Port Plan: `ttnn/cpp/ttnn/operations/reduction/topk`

Port plan for `reduction/topk`, ported from the legacy `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass:** `TopKSingleCoreProgramFactory` only. `TopKMultiCoreProgramFactory` is a
grounded capitulation; Metal 2.0 cannot express its cross-node SRAM-address dependency. The
analysis is in [Deferred / Flagged](#deferred--flagged) and the capitulation is written up as a
Handoff point in `METAL2_PORT_REPORT.md`. The device operation keeps both factories in its
`program_factory_t` variant, on different concepts, and the framework dispatches per factory.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories declare
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`,
  `device/topk_device_operation.hpp:25-36`)
- Variants: two program factories in one `program_factory_t` variant
  (`device/topk_device_operation.hpp:38`), `TopKSingleCoreProgramFactory` (ported here) and
  `TopKMultiCoreProgramFactory` (capitulated). Neither factory has internal variants: each builds
  exactly one program.
- Custom `compute_program_hash`: none, already the default reflection-based hash. Confirmed by
  grep of the op directory, and by the audit's TTNN readiness row.

---

### Variant: TopKSingleCoreProgramFactory  (`device/topk_single_core_program_factory.cpp`)

Despite the name, this factory spreads `Ht` rows across several cores via `split_work_to_cores`
(`:70`); every core in the resulting range runs reader + writer + compute.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_create_index_tensor.cpp` | `core_range` (`:207`) | 0 `input_cb_index`, 1 `index_cb_index`, 2 `Ht`, 3 `Wt`, 4 `total_number_of_cores`, 5 `uint16_output`, then `TensorAccessorArgs(input)` and, only when `tensor_args.indices.has_value()`, `TensorAccessorArgs(indices)` (`:186-197`) | none | per core: 0 `input_tensor` (address), 1 `id`, 2 `work_per_core`, 3 `indices` tensor address or `0u` (`:260-271`) | none | `GENERATE_INDICES = "1"` (`:198-201`) | absent → resolves to `O2` (DM) | `ReaderConfigDescriptor{}` (`:210`) |
| writer | `device/kernels/dataflow/writer_binary_interleaved.cpp` | `core_range` (`:226`) | 0 `output_val_cb_index`, 1 `output_ind_cb_index`, 2 `Ht`, 3 `Ktiles`, 4 `total_number_of_cores`, then `TensorAccessorArgs(values)`, `TensorAccessorArgs(index)` (`:212-220`) | none | per core: 0 `value_tensor` (address), 1 `index_tensor` (address), 2 `id`, 3 `work_per_core` (`:272-279`) | none | none | absent → resolves to `O2` (DM) | `WriterConfigDescriptor{}` (`:228`) |
| compute | `device/kernels/compute/topk.cpp` | `core_range` (`:249`) | 0-7 the eight CB indices `c_0`…`c_7`, 8 `Ht`, 9 `Wt`, 10 `Ktiles`, 11 `largest`, 12 `stable` (`:230-244`) | none | per core: 0 `work_per_core` (`:280-284`) | none | none | absent → resolves to **`O3`** (compute) | `ComputeConfigDescriptor{.fp32_dest_acc_en = !uint16_output || is_fp32_input, .dst_full_sync_en = false, .unpack_to_dest_mode = <UnpackToDestFp32 on c_0/c_2/c_4 when fp32>}` |

`opt_level` is absent from every `KernelDescriptor` in this factory (`grep -n opt_level` over the op
directory returns only the two audit artifacts). Per the resolution rule that still means `O2` on the
two DM kernels and `O3` on the compute kernel.

#### CBs

All eight are declared over the same `core_range`, `total_size = <tile count> * <page size>`, and a
single-element `format_descriptors` with no `tile` field set.

| index | name in factory | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|---|
| `c_0` | `input_cb_index` (`:94-103`) | `input_cb_tile_count (=2) * input_tile_size` | `core_range` | `input_cb_data_format` | `input_tile_size` | not set |
| `c_1` | `index_cb_index` (`:105-114`) | `input_cb_tile_count (=2) * index_tile_size` | `core_range` | `output_ind_cb_data_format` | `index_tile_size` | not set |
| `c_2` | `transposed_val_cb_index` (`:118-127`) | `transposed_cb_tile_count (=4) * compute_tile_size` | `core_range` | `compute_cb_data_format` | `compute_tile_size` | not set |
| `c_3` | `transposed_ind_cb_index` (`:129-138`) | `transposed_cb_tile_count (=4) * index_tile_size` | `core_range` | `output_ind_cb_data_format` | `index_tile_size` | not set |
| `c_4` | `result_prep_val_cb_index` (`:141-150`) | `result_prep_cb_tile_count (=2*Ktiles) * compute_tile_size` | `core_range` | `compute_cb_data_format` | `compute_tile_size` | not set |
| `c_5` | `result_prep_ind_cb_index` (`:152-161`) | `result_prep_cb_tile_count (=2*Ktiles) * index_tile_size` | `core_range` | `output_ind_cb_data_format` | `index_tile_size` | not set |
| `c_6` | `output_val_cb_index` (`:163-172`) | `output_cb_tile_count (=Ktiles) * value_tile_size` | `core_range` | `output_val_cb_data_format` | `value_tile_size` | not set |
| `c_7` | `output_ind_cb_index` (`:174-183`) | `output_cb_tile_count (=Ktiles) * index_tile_size` | `core_range` | `output_ind_cb_data_format` | `index_tile_size` | not set |

No `CBDescriptor` sets `.global_circular_buffer`, `.buffer`, or `.address_offset`, and no
`format_descriptors` list has more than one element, so there is no GlobalCircularBuffer, no
borrowed memory, and no aliasing anywhere in this factory.

#### Semaphores

none; the single-core factory declares no `SemaphoreDescriptor`.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `topk_single_core_program_factory.cpp:194` (`TensorAccessorArgs(input_tensor).append_to`) | `tensor_args.input` | reader RTA 0 (`:263`) |
| `topk_single_core_program_factory.cpp:196` (appended only when `indices.has_value()`) | `tensor_args.indices` (optional) | reader RTA 3 (`:266-270`) |
| `topk_single_core_program_factory.cpp:219` | `std::get<0>(tensor_return_value)` (values output) | writer RTA 0 (`:275`) |
| `topk_single_core_program_factory.cpp:220` | `std::get<1>(tensor_return_value)` (index output) | writer RTA 1 (`:276`) |

Kernel-side consumption, all **Case 1** (address goes straight into a `TensorAccessor`; every access
is through the accessor): `reader_create_index_tensor.cpp:40` (input), `:33` (indices, inside
`#if not GENERATE_INDICES`), `writer_binary_interleaved.cpp:30` (values), `:31` (index). Every
`TensorAccessor` takes exactly two constructor arguments, so there is no third (page-size) argument
to drop.

#### Work split
- Driver: `split_work_to_cores(args.sub_core_grids, Ht, /*row_wise=*/true)` (`:70`)
- num_cores: `total_number_of_cores`
- core_group_1: `core_group_1`, count_per_core: `num_tiles_per_core_group_1`
- core_group_2: `core_group_2`, count_per_core: `num_tiles_per_core_group_2`
- The per-group count reaches the kernels as an **RTA** (`work_per_core`), not a CTA, so the legacy
  factory builds a *single* `KernelDescriptor` per kernel; there is no per-group `KernelDescriptor`
  multiplicity to preserve.

---

### Variant: TopKMultiCoreProgramFactory  (`device/topk_multi_core_program_factory.cpp`)

Inventoried for completeness; **not ported** (see [Deferred / Flagged](#deferred--flagged)). Two node
classes: `local_cores_range_set` (`:129`) and `final_cores_range_set` (`:134`), unioned as
`all_cores_range_set` (`:139-140`).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader_local | `kernels/dataflow/reader_create_index_local_topk.cpp` | local (`:357`) | `c_0`, `c_1`, `Ht`, `Wt_local`, `Wt`, + `TensorAccessorArgs(input)`, `TensorAccessorArgs(indices)` (`:338-346`) | input address, `start_ht`, `start_wt`, `is32_bit_data`, indices address or `0u` (`:500-512`) | `GENERATE_INDICES = indices.has_value() ? "0" : "1"` (`:347-349`) | absent → `O2` | `ReaderConfigDescriptor{}` |
| reader_final | `kernels/dataflow/reader_final_topk.cpp` | final (`:385`) | receiver/sender sem ids, four NoC coords, `Ht`, `Wt_final`, `num_cores-1`, `c_4`, `c_5` (`:367-379`) | none | none | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer_local | `kernels/dataflow/writer_local_topk.cpp` | local (`:410`) | receiver/sender sem ids, final-core NoC x/y, `Ht`, `k`, `Kt`, `c_8`, `c_9`, `c_4`, `c_5` (`:393-404`) | `core_id` (`:515-519`) | none | absent → `O2` | `WriterConfigDescriptor{}` |
| writer_final | `kernels/dataflow/writer_final_topk.cpp` | final (`:430`) | `c_8`, `c_9`, `Ht`, `Kt`, + `TensorAccessorArgs(values)`, `TensorAccessorArgs(index)` (`:417-424`) | values address, index address (`:533-538`) | none | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_local | `kernels/compute/topk_local.cpp` | local (`:458`) | `c_0`, `c_1`, `c_2`, `c_3`, `c_8`, `c_9`, `Ht`, `Wt_local`, `k`, `Kt`, `log2(k)`, `log2(Wt_local)`, `largest`, `sorted`, `stable` (`:437-453`) | `ascending` (`:522-526`) | none | absent → **`O3`** | `ComputeConfigDescriptor{.dst_full_sync_en = false}` |
| compute_final | `kernels/compute/topk_final.cpp` | final (`:488`) | `c_4`, `c_5`, `c_6`, `c_7`, `c_8`, `c_9`, `Ht`, `Wt_final`, `k`, `Kt`, `log2(k)`, `log2(Wt_final)`, `largest`, `sorted`, `stable` (`:467-483`) | none | none | absent → **`O3`** | `ComputeConfigDescriptor{.dst_full_sync_en = false}` |

#### CBs

Ten `CBDescriptor`s over nine indices. `c_8` is declared **twice**, over disjoint ranges with
different data formats. Declaration order is essential: the factory documents at `:158-168` that
it allocates the all-cores CBs first so shared buffers land at a common address before the
core-specific ones.

| index | name | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|---|
| `c_0` | `input_cb_index` (`:171-179`) | `cb_in_units (=4) * value_tile_size` | all | `input_cb_data_format` | `input_tile_size` |
| `c_1` | `index_cb_index` (`:183-191`) | `cb_in_units * index_tile_size` | all | `index_cb_data_format` | `index_tile_size` |
| `c_4` | `gathered_values_cb_index` (`:200-208`) | `Wt_final * compute_tile_size` | all | `compute_cb_data_format` | `compute_tile_size` |
| `c_5` | `gathered_indices_cb_index` (`:212-220`) | `Wt_final * index_tile_size` | all | `index_cb_data_format` | `index_tile_size` |
| `c_9` | `output_ind_cb_index` (`:224-232`) | `num_cb_unit (=2) * index_tile_size` | all | `index_cb_data_format` | `index_tile_size` |
| `c_2` | `input_transposed_cb_index` (`:239-247`) | `Wt_local * compute_tile_size` | local | `compute_cb_data_format` | `compute_tile_size` |
| `c_3` | `index_transposed_cb_index` (`:251-259`) | `Wt_local * index_tile_size` | local | `index_cb_data_format` | `index_tile_size` |
| `c_8` | `values_cb_index` (local copy, `:273-281`) | `num_cb_unit * compute_tile_size` | local | `compute_cb_data_format` | `compute_tile_size` |
| `c_8` | `values_cb_index` (final copy, `:283-291`) | `num_cb_unit * value_tile_size` | final | `value_cb_data_format` | `value_tile_size` |
| `c_6` | `final_values_cb_index` (`:297-305`) | `Wt_final * compute_tile_size` | final | `compute_cb_data_format` | `compute_tile_size` |
| `c_7` | `final_indices_cb_index` (`:309-317`) | `Wt_final * index_tile_size` | final | `index_cb_data_format` | `index_tile_size` |

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`sender_semaphore_id`) | `WORKER` | `all_cores_range_set` | `INVALID` |
| 1 (`receiver_semaphore_id`) | `WORKER` | `all_cores_range_set` | `INVALID` |

Both are plain `SemaphoreDescriptor`s (`:322-333`); no `GlobalSemaphore`.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `topk_multi_core_program_factory.cpp:345` | `tensor_args.input` | reader_local RTA 0 (`:503`) |
| `topk_multi_core_program_factory.cpp:346` | `tensor_args.indices` (optional) | reader_local RTA 4 (`:507-511`) |
| `topk_multi_core_program_factory.cpp:423` | values output | writer_final RTA 0 (`:536`) |
| `topk_multi_core_program_factory.cpp:424` | index output | writer_final RTA 1 (`:537`) |

All Case 1; no third accessor argument anywhere.

#### Work split
- Driver: `cores_utilized(...)` → `find_topk_core_config` (`:111-120`), then
  `select_contiguous_range_from_corerangeset` for the local cores (`:126`) and
  `select_from_corerangeset` for the single final core (`:134-136`).
- num_cores: `num_cores` (local cores + 1 final core); per-local-core width offset `core_id * Wt_local`
  is delivered as an RTA, and the alternating `ascending` flag likewise, so again no per-group CTA
  multiplicity.

---

### Shared kernels

`device/kernels/compute/topk_common_funcs.hpp` is **lent**: this op owns it and two other ops
`#include` it ,
`ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp:13`
and
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp:24`.

No `_metal2` fork exists beside it, and **none is needed or created**: every function in the header
takes its buffer index as a plain `uint32_t` parameter, and a `dfb::name` token carries a constexpr
`operator uint32_t()`, so named tokens pass into the existing signatures unchanged. The header is
also only reached from the multi-core compute kernels, which this pass does not port, so this port
does not touch it at all. The two consumers are a sunset/coordination list, not authorization to
convert in place.

`device/kernels/dataflow/topk_dataflow_common.hpp` is private to this op; both includers are topk
readers (`reader_create_index_tensor.cpp:5`, `reader_create_index_local_topk.cpp:5`). Its
`generate_index_tile(const uint32_t dfb_id, …)` takes a buffer index the same `uint32_t` way, so the
ported single-core reader passes `dfb::index` into it with the header untouched. Not a shared-kernel
Caution case: it is not bound as a kernel source by anything, and its only consumers are inside this
op directory.

No kernel source in this op is bound by any factory outside the op, and no kernel source is shared
*between* the two factories; the nine sources partition cleanly, three to single-core and six to
multi-core. So there is no intra-op shared-kernel coupling either, and the single-core port needs no
fork.

Census run and disambiguated for the three sources this pass converts, per the catalog's Caution
entry. `grep -rl` returns out-of-op hits for two of the filenames; all are false positives:

| filename | out-of-op hit | verdict |
|---|---|---|
| `reader_create_index_tensor.cpp` | `reduction/moe/device/moe_program_factory.cpp:234` | **not a consumer**: `moe` binds its own private copy at `reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp`; same filename, different path |
| `writer_binary_interleaved.cpp` | none | sole consumer is this op's single-core factory |
| `topk.cpp` | `reduction/sources.cmake:43`; `experimental/deepseek_prefill/moe_grouped_topk/sources.cmake:6`; `.../moe_grouped_topk_program_factory.cpp:227,287,345` | **not consumers**; two build files (and the `reduction` one names the *host* file `topk/topk.cpp`, not the compute kernel), plus substring matches on that op's own `moe_grouped_topk.cpp` / `reader_moe_grouped_topk.cpp` |

### Flags

- No unreferenced kernel files: all nine sources under `device/kernels/` are bound by one of the two
  factories.
- `topk.cpp` / `topk.hpp` at the op root are the composite front end (transpose / pad / slice around
  the device op). They carry no kernels, buffers, or bindings and are not part of the port.
- `topk_single_core_program_factory.cpp:79` declares `cores` (`corerange_to_cores(...)`) and never
  uses it; the per-core loop below walks `group.ranges()` instead. Pre-existing dead local; left in
  place and routed to the port report rather than removed in this diff.
- Dead compile-time args, recorded by the audit for the ops team and **ported across as they stand**:
  `Ht` is unused in `reader_create_index_tensor.cpp:23` and `writer_binary_interleaved.cpp:20`;
  `sorted` is unused in `topk_local.cpp:108` and `topk_final.cpp:60` (multi-core, not ported).
- `topk_single_core_program_factory.cpp:198-200` hardcodes `GENERATE_INDICES` to `"1"` (GH issue
  36329), so the reader's precomputed-indices path is compiled out and both the indices RTA and its
  appended `TensorAccessorArgs` are dead today. Ported as written; not "fixed".

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none, already the default reflection-based hash, nothing to
  delete.
- **Implementation notes**:
  - Only `TopKSingleCoreProgramFactory` changes concept. Its declaration in
    `device/topk_device_operation.hpp` becomes
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`;
    `TopKMultiCoreProgramFactory` keeps `create_descriptor`. A `program_factory_t` variant holding
    one factory on each concept is valid and dispatches per factory at runtime.
  - No pybind surface to remove: `topk_nanobind.cpp` binds only the user-facing `ttnn::topk`, never a
    factory entry point.
  - No op-owned tensors: the two outputs come from `create_output_tensors`
    (`device/topk_device_operation.cpp:293-307`), ordinary output allocation.
  - No `override_runtime_arguments` / `get_dynamic_runtime_args` to carry over; the device op has
    neither.

## Planned Spec Shape

*Single-core factory only. `ProgramSpec::name = "topk_single_core"`.*

- **KernelSpecs**: three, 1:1 with the legacy `KernelDescriptor`s, `READER{"reader"}`,
  `WRITER{"writer"}`, `COMPUTE{"compute"}`. No multiplicity: the per-group work count was already an
  RTA in legacy (see Preserved Multiplicity).
- **DataflowBufferSpecs**: eight, 1:1 with the legacy `CBDescriptor`s. `entry_size` = the legacy
  `page_size`, `num_entries` = the legacy tile count (so `entry_size * num_entries` reproduces
  `total_size` exactly), `data_format_metadata` = the legacy `data_format`.
  `tile_format_metadata` stays unset; no legacy `format_descriptors[i].tile` was set.

  | DFBSpecName | legacy CB | entry_size | num_entries | data_format_metadata | endpoints |
  |---|---|---|---|---|---|
  | `INPUT_DFB{"input"}` | `c_0` | `input_tile_size` | `input_cb_tile_count` | `input_cb_data_format` | reader PRODUCER, compute CONSUMER |
  | `INDEX_DFB{"index"}` | `c_1` | `index_tile_size` | `input_cb_tile_count` | `output_ind_cb_data_format` | reader PRODUCER, compute CONSUMER |
  | `TRANSPOSED_VAL_DFB{"transposed_val"}` | `c_2` | `compute_tile_size` | `transposed_cb_tile_count` | `compute_cb_data_format` | compute **self-loop** |
  | `TRANSPOSED_IND_DFB{"transposed_ind"}` | `c_3` | `index_tile_size` | `transposed_cb_tile_count` | `output_ind_cb_data_format` | compute **self-loop** |
  | `RESULT_PREP_VAL_DFB{"result_prep_val"}` | `c_4` | `compute_tile_size` | `result_prep_cb_tile_count` | `compute_cb_data_format` | compute **self-loop** |
  | `RESULT_PREP_IND_DFB{"result_prep_ind"}` | `c_5` | `index_tile_size` | `result_prep_cb_tile_count` | `output_ind_cb_data_format` | compute **self-loop** |
  | `OUTPUT_VAL_DFB{"output_val"}` | `c_6` | `value_tile_size` | `output_cb_tile_count` | `output_val_cb_data_format` | compute PRODUCER, writer CONSUMER |
  | `OUTPUT_IND_DFB{"output_ind"}` | `c_7` | `index_tile_size` | `output_cb_tile_count` | `output_ind_cb_data_format` | compute PRODUCER, writer CONSUMER |

  Endpoint assignments re-derived from the kernel-touch census, not copied from the brief; the
  census agrees with the brief in every case. Every core in the range runs all three kernels, so the
  census is the same on every node: `c_0` / `c_1` have the reader FIFO-producing
  (`reader_create_index_tensor.cpp:54,59,68,72`) and compute FIFO-consuming (`topk.cpp:232,240`) ,
  two touchers, one locked to each role, so 1P+1C with no flag. `c_6` / `c_7` have compute
  FIFO-producing (`topk.cpp:42,49` via `transpose_and_pack`) and the writer FIFO-consuming
  (`writer_binary_interleaved.cpp:45,49`), again 1P+1C. `c_2`…`c_5` are touched by the compute
  kernel alone (one toucher) and it both produces and consumes them, so each is a self-loop, which
  is always legal for a compute kernel. **No DFB needs `allow_instance_multi_binding`**, and no DFB
  is both self-looped and multi-bound.
- **SemaphoreSpecs**: none; the legacy factory declares no semaphores.
- **TensorParameters**: four, one per distinct originating tensor.

  | TensorParamName | tensor | bound by (accessor_name) |
  |---|---|---|
  | `INPUT_TENSOR{"input"}` | `tensor_args.input` | reader (`"input"`) |
  | `INPUT_INDICES_TENSOR{"input_indices"}` | `tensor_args.indices` (optional) | reader (`"indices"`), **only when `indices.has_value()`** |
  | `VALUES_TENSOR{"values"}` | values output | writer (`"values"`) |
  | `INDEX_TENSOR{"index"}` | index output | writer (`"indices"`) |

  Both kernels use the local accessor name `"indices"` for their own indices tensor, accessor names
  are kernel-local, and each kernel's `tensor::indices` resolves to the parameter that kernel binds.
  That keeps both kernels' existing `indices_accessor` local meaningful.

  `TensorSpec` matching stays **strict**: no relaxation is declared. The audit records
  `TensorParameter relaxation = none`, and no topk kernel uses `ArgConfig::Runtime*`.
- **WorkUnitSpecs**: one, `{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_range}`.
  All three kernels and all eight DFBs live on the same node set, so the local-DFB invariant (a DFB's
  producer and consumer share identical work-unit membership) holds trivially.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none; no work-split multiplicity in legacy. `split_work_to_cores` splits `Ht` across two core
groups, but the per-group count reaches all three kernels as the **RTA** `work_per_core`
(`topk_single_core_program_factory.cpp:257-287`), so legacy builds one `KernelDescriptor` per kernel
and the port builds one `KernelSpec` per kernel. Nothing is demoted from CTA to RTA: `work_per_core`
was never a CTA.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `topk_single_core_program_factory.cpp:186` (reader CTA 0) | `input_cb_index` (`tt::CBIndex::c_0`) | `DFBBinding{INPUT_DFB, "input", PRODUCER}` on reader |
| `topk_single_core_program_factory.cpp:186` (reader CTA 1) | `index_cb_index` (`c_1`) | `DFBBinding{INDEX_DFB, "index", PRODUCER}` on reader |
| `topk_single_core_program_factory.cpp:186-193` (reader CTAs 2-5) | positional `Ht`, `Wt`, `total_number_of_cores`, `uint16_output` | named CTAs `"Ht"`, `"Wt"`, `"total_number_of_cores"`, `"uint16_output"` |
| `topk_single_core_program_factory.cpp:194` | `TensorAccessorArgs(input_tensor).append_to(reader_cta)` | `TensorBinding{INPUT_TENSOR, "input"}`; kernel builds `TensorAccessor(tensor::input)` |
| `topk_single_core_program_factory.cpp:196` | `TensorAccessorArgs(indices).append_to(reader_cta)` (conditional) | `TensorBinding{INPUT_INDICES_TENSOR, "indices"}` (conditional); kernel builds `TensorAccessor(tensor::indices)` |
| `topk_single_core_program_factory.cpp:263` (reader RTA 0) | `input_tensor` address via `emplace_runtime_args` | `TensorBinding` above (address auto-injected) |
| `topk_single_core_program_factory.cpp:266-270` (reader RTA 3) | optional `indices` address, or `0u` sentinel | `TensorBinding` above; the `0u` sentinel disappears with the slot |
| `topk_single_core_program_factory.cpp:212` (writer CTA 0) | `output_val_cb_index` (`c_6`) | `DFBBinding{OUTPUT_VAL_DFB, "values", CONSUMER}` on writer |
| `topk_single_core_program_factory.cpp:212` (writer CTA 1) | `output_ind_cb_index` (`c_7`) | `DFBBinding{OUTPUT_IND_DFB, "indices", CONSUMER}` on writer |
| `topk_single_core_program_factory.cpp:212-218` (writer CTAs 2-4) | positional `Ht`, `Ktiles`, `total_number_of_cores` | named CTAs `"Ht"`, `"Kt"`, `"total_number_of_cores"` |
| `topk_single_core_program_factory.cpp:219-220` | `TensorAccessorArgs(value_tensor)`, `TensorAccessorArgs(index_tensor)` | `TensorBinding{VALUES_TENSOR, "values"}`, `TensorBinding{INDEX_TENSOR, "indices"}` |
| `topk_single_core_program_factory.cpp:275-276` (writer RTAs 0-1) | values / index tensor addresses | the two `TensorBinding`s above |
| `topk_single_core_program_factory.cpp:230-238` (compute CTAs 0-7) | the eight CB indices `c_0`…`c_7` | eight `DFBBinding`s on the compute `KernelSpec` (four of them self-loop pairs) |
| `topk_single_core_program_factory.cpp:239-243` (compute CTAs 8-12) | positional `Ht`, `Wt`, `Ktiles`, `largest`, `stable` | named CTAs `"Ht"`, `"Wt"`, `"output_tiles"`, `"largest"`, `"stable_sort"` |
| `reader_create_index_tensor.cpp:27,31` | `TensorAccessorArgs<6>()` + `next_compile_time_args_offset()` chain | gone, `TensorAccessor(tensor::name)` |
| `writer_binary_interleaved.cpp:23-24` | `TensorAccessorArgs<5>()` + chain | gone, `TensorAccessor(tensor::name)` |

Surviving named RTAs: reader `{"id", "work_per_core"}`, writer `{"id", "work_per_core"}`, compute
`{"work_per_core"}`. No CRTAs, and **no varargs**; every kernel reads a fixed set of distinct
fields at constant indices, so all of them are named args.

`work_per_core` carries the same value on every node of a core group but differs between the two
groups, so it stays an RTA (an RTA→CRTA promotion is not available here, and would change dispatch
semantics regardless).

## Hardware configuration

Diffed field by field against the legacy resolved values.

| kernel | legacy | Metal 2.0 |
|---|---|---|
| reader | `ReaderConfigDescriptor{}` → resolved `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`; the reader default | `create_reader_datamovement_config(device.arch())` |
| writer | `WriterConfigDescriptor{}` → resolved `(RISCV_0, NOC_1, DM_DEDICATED_NOC)`; the writer default | `create_writer_datamovement_config(device.arch())` |
| compute | `ComputeConfigDescriptor{.fp32_dest_acc_en = !uint16_output || is_fp32_input, .dst_full_sync_en = false, .unpack_to_dest_mode = …}`; `math_fidelity`, `math_approx_mode` and `bfp8_pack_precise` left at their legacy defaults | `ComputeGen1Config{.enable_32_bit_dest = !uint16_output || is_fp32_input, .double_buffer_dest = true, .unpack_modes = …}` (Style B; the op sets a Metal `ComputeConfigDescriptor` directly, so the config is built directly and *not* routed through the TTNN helper, whose defaults differ) |

`double_buffer_dest = !dst_full_sync_en = true`; the inverted field. It coincides with the
`ComputeGen1Config` default, and is written explicitly so the inversion is visible at review.
`fpu_math_fidelity`, `sfpu_precision_mode` and `bfp_pack_precision_mode` are left at their
`ComputeGen1Config` defaults, which equal the legacy `ComputeConfigDescriptor` defaults
(`HiFi4`, `math_approx_mode = false` → `Precise`, `bfp8_pack_precise = false` → `Approximate`).

**`unpack_modes`: three entries under an fp32 input, none otherwise.** The validator's forced-entry
rule fires when a compute kernel consumes a `Float32`-formatted DFB while `enable_32_bit_dest` is
true. The op reaches that state exactly when the input dtype is `FLOAT32`, which
`validate_on_program_cache_miss` permits alongside `BFLOAT16` and `BFLOAT8_B`. On that path the
legacy factory sets `unpack_to_dest_mode[i] = UnpackToDestFp32` for `c_0`, `c_2` and `c_4` (the three
value-carrying buffers the compute kernel consumes) and leaves every other entry `Default`. The port
mirrors that exactly, keyed by name instead of by index:

| legacy vector entry | Metal 2.0 |
|---|---|
| `unpack_to_dest_mode[input_cb_index] = UnpackToDestFp32` | `{INPUT_DFB, UnpackMode::UnpackToDest}` |
| `unpack_to_dest_mode[transposed_val_cb_index] = UnpackToDestFp32` | `{TRANSPOSED_VAL_DFB, UnpackMode::UnpackToDest}` |
| `unpack_to_dest_mode[result_prep_val_cb_index] = UnpackToDestFp32` | `{RESULT_PREP_VAL_DFB, UnpackMode::UnpackToDest}` |
| every other index, `Default` | entry omitted (`UnpackToSrc`) |

Those three are also precisely the `Float32` DFBs the compute kernel consumes, so the required-entry
rule and the legacy values agree. `OUTPUT_VAL_DFB` is `Float32` too on that path but is only
*produced* by the compute kernel, so it needs no entry. When the input is not fp32 no DFB carries a
32-bit float format, so the table stays empty and every buffer resolves to `UnpackToSrc`, matching
legacy's all-`Default` vector.

## Compiler options

| kernel | legacy resolved `opt_level` | Metal 2.0 |
|---|---|---|
| reader | `O2` (DM default; field absent) | left at the `CompilerOptions` default `O2`; no action |
| writer | `O2` (DM default; field absent) | left at the `CompilerOptions` default `O2`; no action |
| compute | **`O3`** (`ComputeConfigDescriptor` default; field absent) | **explicit** `.compiler_options = {.opt_level = KernelBuildOptLevel::O3}` |

The reader additionally carries `.compiler_options.defines = {{"GENERATE_INDICES", "1"}}`, the
legacy `KernelDescriptor::defines` moved to its Metal 2.0 home.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `c_2` / `c_3` / `c_4` / `c_5` are compute-only workspaces with a single toucher, so the compute
  `KernelSpec` binds each as both PRODUCER and CONSUMER under one accessor name.
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the mechanism the four self-loops above borrow, shared-accessor-name form, so the kernel keeps one
  `DataflowBuffer` object per buffer.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings),
  applied to a **tensor**: `INPUT_INDICES_TENSOR` and its `TensorBinding` are declared only when
  `tensor_args.indices.has_value()`, mirroring the legacy conditional `TensorAccessorArgs` append.
  The kernel-side gate is the pre-existing `#if not GENERATE_INDICES`; no new define is introduced,
  because the host already emits `GENERATE_INDICES` and the kernel already gates every use of the
  indices accessor on it.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `topk.cpp` passes buffer handles to `transpose_tile`, `pack_tile`, `copy_tile`,
  `reconfig_data_format_srca`, `pack_reconfig_data_format`, `compute_kernel_hw_startup` and to its own
  `uint32_t`-taking file-local helpers; `reader_create_index_tensor.cpp` passes one to
  `generate_index_tile`. All go through the `dfb::name → uint32_t` constexpr conversion with no `.id`
  extraction and no temporary wrappers.
- [Same-FIFO aliasing (one DFB, multiple kernel-side names)](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names):
  `topk.cpp` reaches its workspaces through runtime-selected locals (`dfb0`…`dfb3`,
  `topk.cpp:267-270`) whose values are chosen per insertion-sort case. Each buffer keeps **one**
  `DFBBinding` and one `DataflowBuffer` object; the file-scope `constexpr` names are handle aliases
  of the `dfb::` tokens, so the runtime selection logic is unchanged.

## Deferred / Flagged

### `TopKMultiCoreProgramFactory`, capitulation (the port's one structural finding)

The audit raised two open questions and asked the porter to get them answered. Both resolve from the
framework source, and the second one blocks the factory.

**Question 1, can a `DataflowBufferSpec` keep a declared core range covering nodes where no kernel
binds it?** **No, and the choice does not exist.** `DataflowBufferSpec` has no placement field at
all (`tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:79-136`);
placement is derived as the union of the bound kernels' work-unit nodes
(`tt_metal/impl/metal2_host_api/program_spec.cpp:591-600`). So `c_0` / `c_1` necessarily narrow to
the local cores, because no final-core kernel binds them. This is not the dead-CB case; both are
fully live on every local core; it is simply what derived placement means.

**Question 2, does Metal 2.0 give a DFB one common address across its whole node set?** For a DFB
whose derived node set spans both node classes, **yes**: DFB allocation runs through the same
max-watermark allocator as legacy CBs (`tt_metal/impl/program/program.cpp:1560-1583`; the address is
the maximum region end across the buffer's ranges, then marked on all of them), so a spec covering
local + final nodes lands at one address on all of them, exactly as legacy. The narrowing from
Question 1 does not break that: the final core just carries an unused gap where `c_0` / `c_1` used to
sit, and the shared buffers after them still resolve to the max across both classes.

**But `c_4` / `c_5` cannot be single specs spanning both node classes, and that is what breaks the
factory.** `writer_local_topk.cpp:45-50` learns the final core's destination address by reading the
write pointer of its **own** `c_4` / `c_5` instance, so those buffers must be one spec covering both
node classes. The spec validator rejects every possible role assignment for such a spec:

- On a local node the only toucher is `writer_local`, and it only peeks (`get_write_ptr`), so it must
  hold **both** roles; the census requires exactly one producer and exactly one consumer on every
  node in the footprint (`program_spec.cpp:1355-1390`).
- Making `writer_local` a self-loop then trips the self-loop rule: when any kernel appears on both
  sides, the producer and consumer *kernel sets* must be equal (`program_spec.cpp:1438-1444`). Here
  they cannot be; the final node needs `reader_final` (which does the `reserve_back` / `push_back`,
  `reader_final_topk.cpp:34-57`) on one side and `compute_final` (which does the `wait_front` /
  `pop_front`, `topk_final.cpp:85-86,107,126`) on the other.
- Giving `writer_local` a single role instead leaves its node with zero of the other role, which the
  same per-node census rejects.
- `allow_instance_multi_binding` does not rescue it. It skips the role-uniformity checks
  (`program_spec.cpp:1303-1306`) and relaxes the census upper bound, but the self-loop set-equality
  check at `:1438` runs unconditionally. And this DFB is a two-role-per-node 1P+1C on each node
  anyway, so the flag would be wrong on the merits as well as ineffective.

Splitting `c_4` / `c_5` into a local spec and a final spec satisfies the validator but breaks the
transfer: two specs over disjoint ranges get independent addresses, and the local instance is
allocated after `c_0` / `c_1` while the final instance is not, so the local core's write pointer no
longer names the final core's buffer. The failure mode is silent mis-addressing, not an error.

There is no Metal 2.0 mechanism for a kernel to learn a remote node's DFB address:
`CrossNodeDataflowBufferSpec` is explicitly unimplemented and rejected at validation
(`program_spec.cpp:1454-1459`), and threading the address through an RTA is off-whitelist. Reaching
the address by inserting a padding DFB on the final core to realign the watermarks would be
hand-tuning allocation addresses through a fabricated resource, squarely a stop signal.

So the multi-core factory stops here, per the recipe's *When the discipline doesn't fit*. It stays on
`ProgramDescriptorFactoryConcept` and keeps running; the write-up is the port report's Handoff-points
entry.

### Other new findings during planning

- The recipe's endpoint-census vocabulary has no row for "one toucher, but it is a *remote* node's
  buffer it is addressing"; the multi-core `c_4` / `c_5` case above. The audit flagged the same gap
  from its side (its Recipe note 3). Carried to the port report.
- Nothing else: no descriptor type outside the audit's scan appears in either factory, and the
  single-core factory's structure translated 1:1 with no surprises.
