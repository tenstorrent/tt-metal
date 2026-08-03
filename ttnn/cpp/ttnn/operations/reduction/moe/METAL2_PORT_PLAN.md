# Port Plan — `reduction/moe`

Port plan for `ttnn/cpp/ttnn/operations/reduction/moe`, ported from the legacy `ProgramDescriptor`
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for
review.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `MoeProgramFactory::create_descriptor(const MoeParams&, const MoeInputs&, Tensor&)` returns a `tt::tt_metal::ProgramDescriptor` (`device/moe_program_factory.cpp:18-19`).
- Variants: single. `program_factory_t = std::variant<MoeProgramFactory>` (`device/moe_device_operation.hpp:24`); one core (`CoreRange({0,0},{0,0})`, `moe_program_factory.cpp:27`), interleaved only (sharded output is rejected in `validate_on_program_cache_miss`, `moe_device_operation.cpp:46`). No runtime kernel-source selection.
- Custom `compute_program_hash`: none — already the default reflection-based hash. A grep of the op directory for `compute_program_hash` / `attribute_values` / `to_hash` returns nothing. No device-op-class edit is forced by this port.

*(The Metal 2.0 factory concept this port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section
below.)*

### Kernels

All three sources live in this op's own directory. `opt_level` is absent on all three
`KernelDescriptor`s (`grep -n opt_level device/moe_program_factory.cpp` → no hits), so the resolved
levels below come from the per-kernel-type legacy defaults.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_create_index_tensor.cpp` | `CoreRangeSet{CoreRange({0,0},{0,0})}` | `[0]` `input_cb_index`=`c_0`, `[1]` `index_cb_index`=`c_4`, `[2]` `topk_mask_cb_index`=`c_2`, `[3]` `expert_mask_cb_index`=`c_1`, `[4]` `Ht`, `[5]` `Wt`, `[6]` `k`; then `TensorAccessorArgs(input)`, `TensorAccessorArgs(topk_mask)`, `TensorAccessorArgs(expert_mask)` appended (`moe_program_factory.cpp:226-230`) | none | 3 args at node `(0,0)`, emitted via the `MeshTensor` overload of `emplace_runtime_args` (`moe_program_factory.cpp:239-245`): slot 0 `input`, slot 1 `topk_mask`, slot 2 `expert_mask` | none | none | absent → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_interleaved.cpp` | same | `[0]` `out_cb_index`=`c_11`, `[1]` `Ht`, `[2]` `k`; then `TensorAccessorArgs(out)` appended (`moe_program_factory.cpp:247-248`) | none | 1 arg at node `(0,0)` via the same overload (`moe_program_factory.cpp:257-261`): slot 0 `output` | none | none | absent → **O2** | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/moe.cpp` | same | `[0]` `c_0`, `[1]` `c_2`, `[2]` `c_1`, `[3]` `c_3`, `[4]` `c_4`, `[5]` `c_5`, `[6]` `c_6`, `[7]` `c_7`, `[8]` `c_8`, `[9]` `c_11`, `[10]` `Ht`, `[11]` `Wt`, `[12]` `k`, `[13]` `log2(k)`, `[14]` `log2(Wt)`, `[15]` `c_9`, `[16]` `c_10`, `[17]` `tile_width`, `[18]` `c_12` (`moe_program_factory.cpp:263-282`) | none | none | none | none | absent → **O3** | `ComputeConfigDescriptor{}` — every field left at its default (`math_fidelity=HiFi4`, `fp32_dest_acc_en=false`, `dst_full_sync_en=false`, `unpack_to_dest_mode={}`, `bfp8_pack_precise=false`, `math_approx_mode=false`) |

### CBs

All 13 are plain SRAM allocations on the single core range; none is a `GlobalCircularBuffer`, none
sets `.global_circular_buffer`, `.buffer`, or `address_offset`, and no `format_descriptors` list has
more than one element (so there are no aliased CBs). No `CBFormatDescriptor::tile` is ever set, so
`tile_format_metadata` stays unset on every ported DFB.

| index | name | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|---|
| `c_0` | input | `cb_in_units * input_tile_size` (`cb_in_units` = 4) | `{({0,0},{0,0})}` | `input_cb_data_format` | `input_tile_size` | unset |
| `c_1` | expert_mask | `Wt * expert_mask_tile_size` | same | `expert_mask_cb_data_format` | `expert_mask_tile_size` | unset |
| `c_2` | topk_mask | `topk_mask_cb_units * topk_mask_tile_size` (`topk_mask_cb_units` = `Kt`) | same | `topk_mask_cb_data_format` | `topk_mask_tile_size` | unset |
| `c_3` | scale | `scale_tiles * scalar_tile_size` (`scale_tiles` = 1) | same | `scalar_df` | `scalar_tile_size` | unset |
| `c_4` | index | `cb_in_units * index_tile_size` | same | `index_cb_data_format` (`UInt16`) | `index_tile_size` | unset |
| `c_5` | input_transposed | `Wt * value_tile_size` | same | `input_cb_data_format` | `input_tile_size` | unset |
| `c_6` | index_transposed | `Wt * index_tile_size` | same | `index_cb_data_format` | `index_tile_size` | unset |
| `c_7` | values | `values_and_topk_indices_cb_units * value_tile_size` (`= Ht * Kt`) | same | `value_cb_data_format` (`Float16_b`) | `value_tile_size` | unset |
| `c_8` | output_ind | `values_and_topk_indices_cb_units * index_tile_size` | same | `index_cb_data_format` | `index_tile_size` | unset |
| `c_9` | cur_max | `num_out_tiles * out_tile_size` | same | `out_cb_data_format` | `out_tile_size` | unset |
| `c_10` | cur_sum | `num_out_tiles * out_tile_size` | same | `out_cb_data_format` | `out_tile_size` | unset |
| `c_11` | out | `num_out_tiles * out_tile_size` | same | `out_cb_data_format` | `out_tile_size` | unset |
| `c_12` | masked_input | `2 * input_tile_size` | same | `input_cb_data_format` | `input_tile_size` | unset |

**`c_5` is the one row whose `total_size` is not an integer multiple of its own `page_size` by
construction.** It sizes the region with `value_tile_size` (`tile_size(Float16_b)` = 2048) while its
page size is `input_tile_size`. For the BFLOAT16 input the two coincide; for any wider input dtype
they do not. See [Deferred / Flagged](#deferred--flagged).

### Semaphores

none — a grep for `semaphore` / `Semaphore` across the op directory returns nothing and
`desc.semaphores` is never populated.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `moe_program_factory.cpp:228` (`TensorAccessorArgs(input_tensor).append_to(reader_compile_time_args)`) | `tensor_args.input` | reader slot 0 (`moe_program_factory.cpp:242`) |
| `moe_program_factory.cpp:229` (`TensorAccessorArgs(topk_mask_tensor)`) | `tensor_args.topk_mask` | reader slot 1 (`moe_program_factory.cpp:243`) |
| `moe_program_factory.cpp:230` (`TensorAccessorArgs(expert_mask_tensor)`) | `tensor_args.expert_mask` | reader slot 2 (`moe_program_factory.cpp:244`) |
| `moe_program_factory.cpp:248` (`TensorAccessorArgs(out_tensor)`) | `output_tensor` | writer slot 0 (`moe_program_factory.cpp:260`) |

Kernel-side consumption: `TensorAccessor(s0_args, src_addr)` / `(s1_args, topk_addr)` /
`(s2_args, expert_addr)` at `reader_create_index_tensor.cpp:62`, `:66`, `:70`, and
`TensorAccessor(out_args, dst_addr0)` at `writer_unary_interleaved.cpp:31`. All four are Case 1 — the
base only ever reaches a `TensorAccessor`, never raw arithmetic. No site passes a third (page-size)
constructor argument. The compute kernel touches no tensor memory.

### Work split

n/a — single core. The factory hardcodes `CoreRange({0,0},{0,0})` and calls no
`split_work_to_cores`-style helper.

### Shared kernels

none. All three `kernel_source` paths point inside this op's directory, and a repo-wide grep for
`reduction/moe/device/kernels` across `ttnn/` and `tests/` finds no consumer outside this op — so no
source is borrowed, lent, or shared between sibling factories (there is only one factory). The two
out-of-directory `#include`s (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and
`reduce_helpers_compute.hpp`) are kernel-lib headers, explicitly outside the shared-kernel caution's
scope and outside the porter's writeable surface. Neither has a `_metal2` sibling, and neither needs
one: both donor entry points take the DFB id as a `uint32_t` non-type template parameter, which
`dfb::name`'s `constexpr` conversion satisfies unchanged.

### Flags

none. Every `.cpp` under `device/kernels/` is referenced by the factory, and there are no
unreferenced kernel files in the directory. No descriptor type outside the audit's scan appears.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `MoeProgramFactory::create_descriptor` becomes
  `MoeProgramFactory::create_program_artifacts(const MoeParams&, const MoeInputs&, Tensor&)`
  returning `ttnn::device_operation::ProgramArtifacts`. No op-owned tensors, so
  `ProgramArtifacts::op_owned_tensors` stays defaulted. `moe_nanobind.cpp` binds only the user-facing
  `ttnn::moe` function, so no pybind line references the vanishing `create_descriptor` and no pybind
  edit is forced.

  This factory shares the unity-built `ttnn_op_reduction` target with the already-ported
  `accumulation` and `ema` factories, whose anonymous namespaces already hold `ACCUM_*` / `EMA_*`
  spec-name constants. This port's constants therefore carry a `MOE_` prefix
  ([Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)).

## Planned Spec Shape

1:1 with legacy throughout — three `KernelSpec`s, thirteen `DataflowBufferSpec`s, one
`WorkUnitSpec`, no semaphores.

- **KernelSpecs** (3, one per legacy `KernelDescriptor`):
  - `MOE_READER` → `reader_create_index_tensor.cpp`; `hw_config = create_reader_datamovement_config(arch)`; `compile_time_args = {Ht, Wt, K}`; no runtime args; `opt_level` left at Metal 2.0's `O2` (matches the legacy DM default).
  - `MOE_WRITER` → `writer_unary_interleaved.cpp`; `hw_config = create_writer_datamovement_config(arch)`; `compile_time_args = {Ht, K}`; no runtime args; `opt_level` left at `O2`.
  - `MOE_COMPUTE` → `compute/moe.cpp`; `hw_config = ComputeGen1Config{}` (every field at its default — Style B, since the legacy op sets a Metal `ComputeConfigDescriptor` directly with no TTNN `ComputeKernelConfig` feeding it, and Metal 2.0's Gen1 defaults reproduce the legacy `ComputeConfig` defaults field for field); `compile_time_args = {Ht, Wt, K, logk, logWt, tile_width}`; no runtime args; `compiler_options.opt_level = O3` set **explicitly** (legacy `ComputeConfig` defaults to `O3`, Metal 2.0's type-agnostic `CompilerOptions` to `O2`).
  - No `unpack_modes` entry is required: `enable_32_bit_dest` is `false`, so the Float32-consumer rule does not fire. `bfp_pack_precision_mode` stays at its default, matching the legacy `bfp8_pack_precise = false`.
- **DataflowBufferSpecs** (13, one per legacy `CBDescriptor`): `entry_size` = the legacy `page_size`, `num_entries` = the legacy `total_size / page_size`, `data_format_metadata` = the legacy `data_format`, `tile_format_metadata` unset (the legacy `.tile` field was never set). No `borrowed_from` (no CB is backed by a device buffer) and no `advanced_options` on any spec — no aliasing, no multi-binding.

  | DFB | entry_size | num_entries |
  |---|---|---|
  | `input` | `input_tile_size` | `cb_in_units` |
  | `expert_mask` | `expert_mask_tile_size` | `Wt` |
  | `topk_mask` | `topk_mask_tile_size` | `topk_mask_cb_units` |
  | `scale` | `scalar_tile_size` | `scale_tiles` |
  | `index` | `index_tile_size` | `cb_in_units` |
  | `input_transposed` | `input_tile_size` | `Wt * value_tile_size / input_tile_size` |
  | `index_transposed` | `index_tile_size` | `Wt` |
  | `values` | `value_tile_size` | `values_and_topk_indices_cb_units` |
  | `output_ind` | `index_tile_size` | `values_and_topk_indices_cb_units` |
  | `cur_max` | `out_tile_size` | `num_out_tiles` |
  | `cur_sum` | `out_tile_size` | `num_out_tiles` |
  | `out` | `out_tile_size` | `num_out_tiles` |
  | `masked_input` | `input_tile_size` | 2 |

- **DFB bindings** — re-derived from a kernel-touch census over all three sources, not transcribed. On the single node:

  | DFB | touchers (kernel → what it does) | endpoints declared |
  |---|---|---|
  | `input` | reader `reserve_back`/`push_back`; compute `wait_front`/`pop_front` | reader PRODUCER, compute CONSUMER |
  | `expert_mask` | reader `reserve_back`/`push_back`; compute `wait_front`/`pop_front` | reader PRODUCER, compute CONSUMER |
  | `topk_mask` | reader `reserve_back`/`push_back`; compute `wait_front`/`pop_front` (inside `add_block_bcast_rows_inplace`) | reader PRODUCER, compute CONSUMER |
  | `scale` | **writer** `reserve_back`/`push_back` (inside `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler`); compute `wait_front`, never popped (inside `compute_kernel_lib::reduce`, `WaitUpfrontNoPop`) | writer PRODUCER, compute CONSUMER |
  | `index` | reader `reserve_back` + `get_write_ptr` fill + `push_back` (one toucher — the raw peek is bracketed by the same kernel's FIFO ops, so the PRODUCER binding covers it); compute `wait_front`/`pop_front` | reader PRODUCER, compute CONSUMER |
  | `input_transposed` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |
  | `index_transposed` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |
  | `values` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |
  | `output_ind` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |
  | `cur_max` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |
  | `cur_sum` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |
  | `out` | compute `reserve_back`/`push_back` (inside `compute_kernel_lib::reduce`); writer `wait_front`/`pop_front` | compute PRODUCER, writer CONSUMER |
  | `masked_input` | compute only | compute PRODUCER **and** CONSUMER (self-loop) |

  The census agrees with the brief on all thirteen rows: six legal 1:1, seven self-loops, zero dead,
  zero multi-binding. Both self-loop endpoints share one `accessor_name`, so the compute kernel gets
  one `dfb::name` handle per buffer.
- **SemaphoreSpecs**: none.
- **TensorParameters** (4, one per distinct originating tensor, each with exactly one `TensorBinding`): `MOE_TENSOR_INPUT` → reader `tensor::input`; `MOE_TENSOR_TOPK_MASK` → reader `tensor::topk_mask`; `MOE_TENSOR_EXPERT_MASK` → reader `tensor::expert_mask`; `MOE_TENSOR_OUTPUT` → writer `tensor::output`. All strict (`relaxations` left default) — no relaxation is flagged for this op.
- **WorkUnitSpecs** (1): `{.name = "main", .kernels = {MOE_READER, MOE_WRITER, MOE_COMPUTE}, .target_nodes = NodeCoord{0, 0}}`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. The factory pushes exactly three `KernelDescriptor`s,
one per source, over a single core.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `moe_program_factory.cpp:239-245` → reader RTA slot 0 | `input_tensor` via the `MeshTensor` overload of `emplace_runtime_args`; kernel reads `get_arg_val<uint32_t>(0)` (`reader:41`) and feeds it to `TensorAccessor` (`reader:62`) | `TensorParameter MOE_TENSOR_INPUT` + reader `TensorBinding{.accessor_name = "input"}`; kernel writes `TensorAccessor(tensor::input)` |
| `moe_program_factory.cpp:239-245` → reader RTA slot 1 | `topk_mask_tensor`; kernel `get_arg_val<uint32_t>(1)` (`reader:42`) → `TensorAccessor` (`reader:66`) | `TensorParameter MOE_TENSOR_TOPK_MASK` + reader `TensorBinding{.accessor_name = "topk_mask"}` |
| `moe_program_factory.cpp:239-245` → reader RTA slot 2 | `expert_mask_tensor`; kernel `get_arg_val<uint32_t>(2)` (`reader:43`) → `TensorAccessor` (`reader:70`) | `TensorParameter MOE_TENSOR_EXPERT_MASK` + reader `TensorBinding{.accessor_name = "expert_mask"}` |
| `moe_program_factory.cpp:257-261` → writer RTA slot 0 | `out_tensor`; kernel `get_arg_val<uint32_t>(0)` (`writer:12`) → `TensorAccessor` (`writer:31`) | `TensorParameter MOE_TENSOR_OUTPUT` + writer `TensorBinding{.accessor_name = "output"}` |
| `moe_program_factory.cpp:228` / `:229` / `:230` / `:248` | four `TensorAccessorArgs(<tensor>).append_to(<cta vector>)` calls | the binding mechanism end-to-end — the host packs the layout metadata at program creation |
| `reader:55-57` | `TensorAccessorArgs<7>()` and the chained `next_compile_time_args_offset()` pair | gone with the four bindings above |
| `writer:19` | `TensorAccessorArgs<3>()` | gone |
| reader CTA slots 0-3 (`moe_program_factory.cpp:227`) | `input_cb_index`, `index_cb_index`, `topk_mask_cb_index`, `expert_mask_cb_index`; kernel reads them at `reader:45-48` | `DFBBinding`s on the reader `KernelSpec` (`input`, `index`, `topk_mask`, `expert_mask`, all PRODUCER); kernel writes `DataflowBuffer dfb_in0(dfb::input)` etc. |
| writer CTA slot 0 (`moe_program_factory.cpp:247`) | `out_cb_index`; kernel reads it at `writer:14` | `DFBBinding{out, "out", CONSUMER}`; kernel writes `DataflowBuffer dfb_out(dfb::out)` |
| `writer:27` | `constexpr uint32_t scale_dfb_index = tt::CBIndex::c_3;` — a hardcoded CB index duplicating the host's choice at `moe_program_factory.cpp:106`, with no CTA behind it | `DFBBinding{scale, "scale", PRODUCER}` on the writer `KernelSpec`; the kernel-lib call site becomes `calculate_and_prepare_reduce_scaler<dfb::scale, …>()`. The writer **gains** a DFB binding where the legacy kernel declared none. |
| compute CTA slots 0-9, 15, 16, 18 (`moe_program_factory.cpp:263-282`) | twelve CB indices (`c_0`, `c_2`, `c_1`, `c_3`, `c_4`, `c_5`, `c_6`, `c_7`, `c_8`, `c_11`, `c_9`, `c_10`, `c_12`) read at `compute:439-448`, `:456-457`, `:459` | thirteen `DFBBinding`s on the compute `KernelSpec` (six 1:1 roles plus seven self-loop PRODUCER+CONSUMER pairs); the kernel passes `dfb::name` at every use site |
| reader CTA slots 4-6; writer CTA slots 1-2; compute CTA slots 10-14, 17 | positional CTAs | named CTAs. reader `{"Ht", "Wt", "K"}`; writer `{"Ht", "K"}`; compute `{"Ht", "Wt", "K", "logk", "logWt", "tile_width"}`. Kernels read them via `get_arg(args::<name>)`. |
| `writer:23`, `reader:60`, `:64`, `:68` | `get_tile_size(<cb id>)` — cb-id-keyed free function | the `DataflowBuffer` member getter `dfb.get_tile_size()` (whitelist rule 7). The values stay: they are the *transfer size* argument to `noc.async_read` / `async_write`, not an accessor page-size override. |
| `writer:24` | `const DataFormat data_format = get_dataformat(out_dfb_index);` — cb-id-keyed free function whose result is never used | dropped. Rule 7 forces this line to change and the local is dead, so no member-getter call replaces it. |

Semaphore-ID RTAs: none (the op has no semaphores). Page-size third-argument CTAs/RTAs: none (no
accessor site passes one). Case 2 (raw-pointer) bindings: none. Varargs: none introduced — every
legacy argument read uses a literal constant index and is a distinct field, and all four RTAs become
tensor bindings, leaving the reader and writer with **zero** runtime args (so neither gets a
`KernelRunArgs` entry).

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding) /
  [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  seven compute-internal scratch DFBs (`input_transposed`, `index_transposed`, `values`,
  `output_ind`, `cur_max`, `cur_sum`, `masked_input`), each bound PRODUCER **and** CONSUMER on the
  one compute `KernelSpec` under a shared `accessor_name`.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `dfb::name` flows unwrapped into `transpose_init`, `pack_tile`, `add_tiles_bcast_rows`,
  `reconfig_data_format`, `compute_kernel_hw_startup`, and both kernel-lib entry points — including
  the two that take the id as a `uint32_t` non-type template parameter
  (`compute_kernel_lib::reduce<…>`, `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<…>`).
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  `MOE_`-prefixed spec-name constants, since `ttnn_op_reduction` is unity-built and the sibling
  `accumulation` / `ema` factories already declare `ACCUM_*` / `EMA_*` constants in their anonymous
  namespaces.

Not applied, and why: no conditional/optional bindings (nothing is host-conditional in this
factory), no aliased DFBs (every legacy `format_descriptors` list has exactly one element), no
same-FIFO aliasing (no kernel or host site maps two names onto one CB), no multi-variant branching
(one config), no two-toucher 1P+1C assignment (no DFB has two touchers that aren't already a locked
producer/consumer pair).

## Deferred / Flagged

- **`c_5` input_transposed sizes its region with a different tile size than its page size.**
  `moe_program_factory.cpp:133-141` sets `.total_size = Wt * value_tile_size` (2048 bytes per tile,
  fixed at `Float16_b`) while `.page_size = input_tile_size`. Metal 2.0 has no separate `total_size`
  field, so the faithful translation is `entry_size = input_tile_size` with
  `num_entries = Wt * value_tile_size / input_tile_size` — byte-identical to the legacy allocation
  for every dtype, which preserves the latent under-allocation for a FLOAT32 input rather than
  silently fixing it. Writing `num_entries = Wt` instead would have been the "obviously intended"
  value but would change the FLOAT32 footprint, so it is out of scope. Reported to the ops team as
  audit misc anomaly 1; carried into the port report's Open items.
- **The writer's dead `get_dataformat` local is dropped rather than rewritten.** Whitelist rule 7
  forces `get_dataformat(out_dfb_index)` (`writer:24`) to change because the cb id is gone; the local
  it initializes is never read. Dropping the statement is the zero-functional-change resolution, and
  the audit called it out in advance. The two dead `constexpr uint32_t onetile = 1;` locals
  (`reader:59`, `writer:22`) are *not* port work — nothing about them is a CB-id or argument
  construct — so they are left exactly as they are.
- **A second, newer kernel entry-point convention exists but is not the port target.**
  `tt_metal/hw/inc/experimental/kernel_args.h:44-47` defines a `TT_KERNEL` marker from which the JIT
  generates `kernel_main()` out of a tagged function's signature, and
  `tt_metal/jit_build/kernel_signature_parser.hpp` documents it as fully optional (a source with no
  marker keeps its hand-written `kernel_main()`). The recipe prescribes the
  `void kernel_main()` + `get_arg(args::name)` form, so this port uses that. Noted so a later reader
  does not read the absence of `TT_KERNEL` as an oversight.
- No new structural findings the audit missed. No feature gate fired during planning, no construct
  needed a legacy workaround, and no site required reaching past the op's directory.
