# Port Plan — topk (`TopKSingleCoreProgramFactory`)

Port plan for `ttnn/cpp/ttnn/operations/reduction/topk`, ported from the `ProgramDescriptor`
API (`create_descriptor`) to Metal 2.0 (`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope: `TopKSingleCoreProgramFactory` ONLY.** The sibling `TopKMultiCoreProgramFactory`
is RED (cross-node gather DFB, unsupported) and is NOT touched — it stays on the legacy
`ProgramDescriptorFactoryConcept`. The `program_factory_t` variant legally mixes concepts;
the framework dispatches per-factory.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`),
  `device/topk_single_core_program_factory.cpp:18`.
- Variants: single (the device-op has two *factories*, but the single-core factory itself
  is single-variant — no `reduce_dim`-style branch).
- Custom `compute_program_hash`: none — `TopKDeviceOperation` uses the default reflection hash
  (audit Q5). Nothing to delete.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per-core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_create_index_tensor.cpp` | `core_range` (single core after split) | `cb_id_in0(c0), cb_intermed_index(c1), Ht, Wt, total_number_of_cores, uint16_output` + `TensorAccessorArgs(input)` + `TensorAccessorArgs(indices)` (if present) | `src_addr(=input obj), id, work_per_core, src_indices_addr(=indices.address() or 0)` | `{GENERATE_INDICES:"1"}` (hardcoded, GH #36329) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_binary_interleaved.cpp` | `core_range` | `values_cb(c6), output_ind_cb(c7), Ht, Kt, total_number_of_cores` + `TensorAccessorArgs(value)` + `TensorAccessorArgs(index)` | `dst_addr0(=value obj), dst_addr1(=index obj), id, work_per_core` | none | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/topk.cpp` | `core_range` | `in_val(c0), in_ind(c1), transp_val(c2), transp_ind(c3), prep_val(c4), prep_ind(c5), out_val(c6), out_ind(c7), Ht, Wt, Ktiles, largest` | `work_per_core` | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = !uint16_output, .dst_full_sync_en=false}` |

Note: `Ht` in reader/writer CTAs is *unused* by the kernels (both compute `Ht` and the reader/writer
`Ht` CTA are dead in the single-core kernel). Preserve them verbatim — not the porter's call to remove.

### CBs (all on `core_range`, single format descriptor each — no aliasing)
| index | total_size | data_format | page_size |
|---|---|---|---|
| c_0 input_cb | `2 * input_tile_size` | input dtype | `input_tile_size` |
| c_1 index_cb | `2 * index_tile_size` | index (output ind) dtype | `index_tile_size` |
| c_2 transposed_val_cb | `4 * compute_tile_size` | compute dtype (bf16 if bfp8/4) | `compute_tile_size` |
| c_3 transposed_ind_cb | `4 * index_tile_size` | index dtype | `index_tile_size` |
| c_4 result_prep_val_cb | `2*Ktiles * compute_tile_size` | compute dtype | `compute_tile_size` |
| c_5 result_prep_ind_cb | `2*Ktiles * index_tile_size` | index dtype | `index_tile_size` |
| c_6 output_val_cb | `Ktiles * value_tile_size` | value (output) dtype | `value_tile_size` |
| c_7 output_ind_cb | `Ktiles * index_tile_size` | index dtype | `index_tile_size` |

No `tile` field set on any legacy `CBFormatDescriptor` → no `tile_format_metadata` to carry.

### Semaphores
none.

### Tensor accessors
| host site | originating Tensor | RTA slot (host) | role |
|---|---|---|---|
| reader CTA `TensorAccessorArgs(input)` | input | reader RTA 0 (`src_addr`, object) | Case 1 → `ta::input` |
| reader CTA `TensorAccessorArgs(indices)` (opt) | indices (optional input) | reader RTA 3 (`.address()`) | Case 1 → `ta::indices` |
| writer CTA `TensorAccessorArgs(value)` | values output | writer RTA 0 (`dst_addr0`, object) | Case 1 → `ta::values` |
| writer CTA `TensorAccessorArgs(index)` | indices output | writer RTA 1 (`dst_addr1`, object) | Case 1 → `ta::out_indices` |

All Case 1 (consumed via `TensorAccessor`). No Case 2 raw-pointer bindings. Compute kernel
touches CB L1 only — no tensor binding.

### Work split
- Driver: `split_work_to_cores(args.sub_core_grids, Ht, true)`.
- For the *single-core* factory this resolves to a single core (the program selects one core
  from the grid). `num_tiles_per_core_group_*` feed only `work_per_core`, which is an **RTA**,
  not a CTA. There is **no per-group CTA difference** → no preserved multiplicity.
- Per-core RTAs: `id` (sequential core index, 0-based) and `work_per_core`. Loop emits them
  per core across both groups; for the single-core case this is one core.

### Cross-op kernels
none — all three kernels live in this op's `device/kernels/`.

### Flags
- Shared in-dir headers `topk_dataflow_common.hpp` (used by `reader_create_index_tensor.cpp`
  AND the multi-core `reader_create_index_local_topk.cpp`) and `topk_common_funcs.hpp` (compute
  kernels). **Do not modify** — they are shared with the RED multi-core kernels. `generate_index_tile`
  takes a `uint32_t cb_id`; `dfb::index` converts implicitly, so no header edit is needed.
- `GENERATE_INDICES` is hardcoded to `"1"` (GH #36329), so the `#if not GENERATE_INDICES`
  precomputed-indices read path in `reader_create_index_tensor.cpp` is dead in single-core today.
  The optional `indices` binding is enumerated anyway (per brief).
- Reader/writer `Ht` CTA and compute `Ht` CTA are dead (unused by the kernels). Preserved verbatim.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: single-program, no op-owned tensors, no MeshWorkload, no semaphores.
  The factory header changes the single-core factory's method from `create_descriptor`
  (returning `ProgramDescriptor`) to `create_program_artifacts` (returning
  `ttnn::device_operation::ProgramArtifacts`). The multi-core factory's method is untouched.

## Planned Spec Shape
- **KernelSpecs (3):** reader, writer, compute — 1:1 with legacy `KernelDescriptor`s, same source.
- **DataflowBufferSpecs (8):** one per legacy CB (c_0..c_7). No aliasing, no borrowed memory,
  no tile metadata. `entry_size` = legacy page_size; `num_entries` = legacy total_size/page_size.
- **SemaphoreSpecs:** none.
- **TensorParameters (4):** input, indices (optional), values output, out_indices output.
- **WorkUnitSpecs (1):** one WU {reader, writer, compute} over `core_range` (single core).

### DFB bindings per kernel
- reader: c_0 PRODUCER ("in0"), c_1 PRODUCER ("index").  (reader fills both input value and
  the generated index CB.)
- compute: c_0 CONSUMER, c_1 CONSUMER, c_2 PROD+CONS (self-loop), c_3 PROD+CONS (self-loop),
  c_4 PROD+CONS (self-loop), c_5 PROD+CONS (self-loop), c_6 PRODUCER, c_7 PRODUCER.
  c_2..c_5 are scratch/staging buffers the compute kernel both reserves/pushes and waits/pops
  within itself → genuine self-loop (real producer AND consumer, not a fake-CB).
- writer: c_6 CONSUMER ("values"), c_7 CONSUMER ("indices").

### Tensor bindings per kernel
- reader: `ta::input` (input param), `ta::indices` (indices param, optional — bound only when
  `tensor_args.indices.has_value()`, gated by `GENERATE_INDICES`... see note below).
- writer: `ta::values` (values output param), `ta::out_indices` (out_indices output param).

## Preserved Multiplicity
none — no work-split CTA multiplicity in legacy (per-group counts are RTAs).

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA 0,1 | `cb_id_in0`, `cb_intermed_index` (CB indices) | `DFBBinding` (c_0, c_1) |
| reader CTA `TensorAccessorArgs(input)` + RTA 0 (`src_addr` object) | TA args + addr-RTA | `TensorBinding` `ta::input` |
| reader CTA `TensorAccessorArgs(indices)` + RTA 3 (`.address()`) | TA args + addr-RTA | `TensorBinding` `ta::indices` (optional) |
| reader CTAs 2,3,4,5 | `Ht,Wt,total_cores,uint16_output` positional | named CTAs `args::Ht` etc. |
| reader RTAs 1,2 | `id, work_per_core` positional | named RTAs `args::id`, `args::work_per_core` |
| writer CTA 0,1 | `values_cb`, `output_ind_cb` (CB indices) | `DFBBinding` (c_6, c_7) |
| writer CTA `TensorAccessorArgs(value/index)` + RTAs 0,1 | TA args + addr-RTAs | `TensorBinding` `ta::values`, `ta::out_indices` |
| writer CTAs 2,3,4 | `Ht,Kt,total_cores` positional | named CTAs |
| writer RTAs 2,3 | `id, work_per_core` positional | named RTAs |
| compute CTAs 0..7 | 8 CB indices | `DFBBinding`s (c_0..c_7) |
| compute CTAs 8..11 | `Ht,Wt,Ktiles,largest` positional | named CTAs |
| compute RTA 0 | `work_per_core` positional | named RTA `args::work_per_core` |

## Applied Patterns
- [Self-loop DFB binding](metal2_port_patterns.md#pattern-self-loop-dfb-binding): c_2,c_3,c_4,c_5
  on the compute KernelSpec — PRODUCER + CONSUMER both. These are real staging/scratch buffers
  the compute kernel both fills and drains within itself (transpose staging + result-prep
  double buffer), so the producer and consumer are genuine (not fake-CB self-loops).
- [Conditional / optional binding](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings):
  `ta::indices` on the reader, gated. See Deferred/Flagged for the GENERATE_INDICES interaction.
- [Pass DFB handles directly to LLKs](metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  compute kernel passes `dfb::*` directly to `transpose_wh_*`, `pack_tile`, `copy_tile`, etc.;
  reader passes `dfb::index` to the shared `generate_index_tile(uint32_t, ...)` helper.

## Deferred / Flagged
- **GENERATE_INDICES / optional-indices interaction (GH #36329).** The legacy factory hardcodes
  `GENERATE_INDICES=1`, so the precomputed-indices read path (the only consumer of `ta::indices`)
  is compiled out. The brief says to bind the indices param anyway. Decision for the port: the
  reader's `ta::indices` binding and the `src_indices_addr` read are gated by the *same*
  `#if not GENERATE_INDICES` the kernel already uses. Since GENERATE_INDICES is always 1 today,
  the binding is omitted at the host whenever the define says generate (i.e. always, today). To
  keep the typed path correct and future-proof for when #36329 is fixed, the host binds
  `ta::indices` only when `tensor_args.indices.has_value() && !generate_indices`, and emits the
  `GENERATE_INDICES` define so the kernel `#if not GENERATE_INDICES` gate matches the binding
  presence. See PORT_REPORT for the friction note.
