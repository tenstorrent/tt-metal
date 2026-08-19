# Port Plan — `moreh/moreh_linear_backward` (`MorehBiasAddBackwardOperation`)

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward`, ported from the
`ProgramDescriptor` host API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this change: both factories.** `SingleCoreProgramFactory` and `MultiCoreProgramFactory`
share `device/kernels/writer_moreh_bias_backward.cpp`, so converting one alone would break the
other. Co-converting both keeps the shared writer at rung 0 (no `_metal2` fork). The op is small
enough that this is comfortably one pass.

The op directory holds **one** device operation. The op's input- and weight-gradient paths are not
device operations — `ttnn::moreh_linear_backward` composes them from `ttnn::moreh_matmul` /
`ttnn::moreh_sum`, which live in their own directories and are out of scope.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor()` returning
  `tt::tt_metal::ProgramDescriptor`, on both factories
  (`device/moreh_linear_backward_device_operation.hpp:33,40`).
- Variants: **two** factories, both nested structs inside the device-operation class and both
  members of the existing `program_factory_t` variant
  (`device/moreh_linear_backward_device_operation.hpp:46`). Selected by `is_scalar(bias_grad)` in
  `select_program_factory` (`device/moreh_linear_backward_device_operation.cpp:28-35`):
  scalar `bias_grad` → `SingleCoreProgramFactory`, 1-D `bias_grad` → `MultiCoreProgramFactory`.
  **Not** the direct-descriptor shape, so `ttnn_factory.md` exception 3 does not apply.
- Custom `compute_program_hash`: **none** — default reflection-based hash, and no backdoor
  `attribute_values` / `to_hash`. Nothing to leave alone.
- `override_runtime_arguments`: **none** (this is what selects the base concept).
- `get_dynamic_runtime_args`: **none**.
- Runtime kernel-source selection: **none** — each `KernelDescriptor` has one fixed source.

### Variant: `SingleCoreProgramFactory`

`device/moreh_linear_backward_single_core_program_factory.cpp`. All resources sit on the single
core `core_set = CoreRangeSet{CoreRange({0,0},{0,0})}`.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_bias_backward_hw.cpp` | `core_set` | `TensorAccessorArgs(*output_grad.buffer()).get_compile_time_args()` (`:128-129`) — accessor plumbing **only**, no other CTA | none | `{output_grad.buffer(), num_tiles, 0u, mask_h, mask_w, do_mask_h, do_mask_w}` on `{0,0}` (`:180-188`) | none | none | absent → **O2** | `ReaderConfigDescriptor{}` (`:139`) |
| writer | `device/kernels/writer_moreh_bias_backward.cpp` | `core_set` | `TensorAccessorArgs(*bias_grad.buffer()).get_compile_time_args()` (`:130-131`) — accessor plumbing only | none | `{bias_grad.buffer(), 1u, 0u}` (`:189`) | none | none | absent → **O2** | `WriterConfigDescriptor{}` (`:147`) |
| compute | `device/kernels/moreh_bias_backward_single_core_hw.cpp` | `core_set` | `{}` — empty (`:167`) | none | `{batch_num, Ht, Wt, do_mask_h, do_mask_w}` (`:190-191`) | none | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_SCALAR`, + `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` (`:152-159`) | absent → **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:169-175`) |

`grep -n opt_level` over `device/` returns **zero** hits, so every resolved level above is the
legacy per-kernel-type default: `O2` for the two DM descriptors, **`O3`** for the
`ComputeConfigDescriptor`.

`unpack_to_dest_mode` is `std::vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, Default)` and is
**never modified** in this factory (`:160`) — every entry stays `Default`.

#### CBs
All `core_ranges = core_set`; `page_size == total_size / num_tiles`; no `tile` field set on any
format descriptor; every `format_descriptors` list is single-element (no aliasing); no
`GlobalCircularBuffer`, no `.buffer` (borrowed memory), no `address_offset`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | note |
|---|---|---|---|---|---|---|
| `c_0` | `in0_t(2) * tile_size(cb_data_format)` | `core_set` | `cb_data_format` | `tile_size(cb_data_format)` | — | output_grad tiles (`:68-76`) |
| `c_1` | `in1_t(1) * tile_size(cb_data_format)` | `core_set` | `cb_data_format` | same | — | scaler (`:77-85`) |
| `c_2` | `in2_t * tile_size(cb_data_format)`, `in2_t = (do_mask_h \|\| do_mask_w) ? 2 : 0` | `core_set` | `cb_data_format` | same | — | mask_h_w — **allocated only when `in2_t > 0`** (`:43,86-96`) |
| `c_16` | `out0_t(1) * tile_size(cb_data_format)` | `core_set` | `cb_data_format` | same | — | bias_grad out (`:97-105`) |
| `c_24` | `im0_t(1) * tile_size(cb_data_format)` | `core_set` | `cb_data_format` | same | — | intermed0 (`:106-114`) |
| `c_25` | `im1_t(1) * tile_size(fp32_dest_acc_en_data_format)` | `core_set` | `fp32_dest_acc_en_data_format` = `fp32_dest_acc_en ? Float32 : cb_data_format` | `tile_size(fp32_dest_acc_en_data_format)` | — | intermed1 / accumulator (`:64,115-123`) |

#### Semaphores
none — `grep -i semaphore` over the op directory returns zero hits.

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._single_core_program_factory.cpp:128-129` | `tensor_args.output_grad` | reader RTA slot 0 (`output_grad.buffer()`, `:182`) |
| `..._single_core_program_factory.cpp:130-131` | `bias_grad` (`tensor_return_value_t`) | writer RTA slot 0 (`bias_grad.buffer()`, `:189`) |

Kernel-side: `reader_moreh_bias_backward_hw.cpp:20,39` (`TensorAccessorArgs<0>()` →
`TensorAccessor(src_args, src_addr)`), `writer_moreh_bias_backward.cpp:16,20`
(`TensorAccessor(dst_args, dst_addr)`). Both **Case 1** — the base flows only into a
`TensorAccessor` and is accessed by `{.page_id = …}`; neither passes a third (page-size) argument.
`tensor_args.bias` is read on the host only (for the output spec) and is **not** a binding.

#### Work split
n/a — single core `{0,0}`.

### Variant: `MultiCoreProgramFactory`

`device/moreh_linear_backward_multi_core_program_factory.cpp`.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_bias_backward_h.cpp` | `all_cores` | `TensorAccessorArgs(*output_grad.buffer()).get_compile_time_args()` (`:132-133`) — accessor plumbing only | none | per-core `{output_grad_buf, num_tiles, Wt, num_cols_per_core, tile_offset, mask_h, mask_w, do_mask_h, do_mask_w && core_has_last_wt}` (`:217-227`) | none | none | absent → **O2** | `ReaderConfigDescriptor{}` (`:143`) |
| writer | `device/kernels/writer_moreh_bias_backward.cpp` | `all_cores` | `TensorAccessorArgs(*bias_grad.buffer()).get_compile_time_args()` (`:134-135`) | none | per-core `{bias_grad_buf, num_cols_per_core, tile_offset}` (`:229`) | none | none | absent → **O2** | `WriterConfigDescriptor{}` (`:151`) |
| compute_desc_1 | `device/kernels/moreh_bias_backward_multi_core_h.cpp` | `core_group_1` | `{num_cols_per_core_group_1}` (`:171`) — **dead** (see Flags) | none | per-core in group 1 `{batch_num, Ht, num_cols_per_core, do_mask_h, do_mask_w && core_has_last_wt}` (`:232-238`) | none | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_COL`, + `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` (`:156-164`) | absent → **O3** | `ComputeConfigDescriptor{…}` (`:173-179`) |
| compute_desc_2 | same source | `core_group_2`, only when `has_core_group_2` (`:182-197`) | `{num_cols_per_core_group_2}` (`:188`) — **dead** | none | per-core in group 2, same five names (`:241-247`) | none | same | absent → **O3** | `ComputeConfigDescriptor{…}` (`:190-196`) |

`unpack_to_dest_mode` starts all-`Default` and sets **`unpack_to_dest_mode[CBIndex::c_25] =
UnpackToDestFp32` when `fp32_dest_acc_en`** (`:159-163`). Both compute descriptors share the same
vector.

#### CBs
Same six CBs, same formats and sizes as the single-core factory, all with
`core_ranges = all_cores`. **One difference:** `c_2` is allocated **unconditionally** with
`in2_t = 2` (`:64,92-100`) — no `do_mask_h || do_mask_w` guard.

#### Semaphores
none.

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._multi_core_program_factory.cpp:132-133` | `tensor_args.output_grad` | reader RTA slot 0 (`output_grad_buf`, `:219`) |
| `..._multi_core_program_factory.cpp:134-135` | `bias_grad` | writer RTA slot 0 (`bias_grad_buf`, `:229`) |

Kernel-side: `reader_moreh_bias_backward_h.cpp:22,35` and
`writer_moreh_bias_backward.cpp:16,20`. Both **Case 1**, no third argument.

#### Work split
- Driver: `split_work_to_cores(grid, Wt)` (`:51-57`), `grid = device->compute_with_storage_grid_size()`
- `num_cores_to_be_used`, `all_cores`, `core_group_1`, `core_group_2`,
  `num_cols_per_core_group_1`, `num_cols_per_core_group_2`
- Per-core iteration: `CoreCoord core = {i / num_cores_y, i % num_cores_y}` with
  `num_cores_y = grid.y` (`:48,205`); `tile_offset` accumulates `num_cols_per_core` (`:251`)

### Shared kernels
| kernel | class | `_metal2` fork beside it? | rung |
|---|---|---|---|
| `device/kernels/writer_moreh_bias_backward.cpp` | **intra-op** — bound by `SingleCoreProgramFactory` (`:143`) and `MultiCoreProgramFactory` (`:147`) | no | **rung 0 — no fork.** Both binding factories convert in this change, so the file is Metal-2.0-ified in place with no consumer left behind. |

Census run per kernel with `grep -rl <filename> ttnn/cpp/ttnn/operations/`, hits disambiguated:
for all five kernels the only hits are this op's own two factories (plus this port's own
`METAL2_*.md` files). **No borrowed kernels** (all five live in this op's `device/kernels/`) and
**no lent kernels** (nothing outside `moreh_linear_backward` binds any of them), so the port
creates no cross-op coordination cost and no sunset list.

Donor headers the kernels call into (`ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`,
`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_{dataflow,compute}.hpp`) are **not** shared kernels —
they are helper pools crossed by the `dfb::name → uint32_t` conversion and are left untouched.
`fill_cb_with_value` and `generate_mask_h_w` already take a `DataflowBuffer` by value.

### Flags
- **Dead compile-time arg on both multi-core compute descriptors.**
  `..._multi_core_program_factory.cpp:171` and `:188` set
  `compile_time_args = {num_cols_per_core_group_N}`, but
  `moreh_bias_backward_multi_core_h.cpp` reads **no** compile-time argument
  (`grep -n get_compile_time_arg_val` over all five kernels → zero hits). The per-core count the
  kernel actually uses arrives as RTA slot 2 (`Wt_per_core`, `..._multi_core_h.cpp:14`), fed from
  the same `num_cols_per_core`. **Preserved verbatim as a named CTA** — dropping it would collapse
  the per-group multiplicity, which is an owner decision, not port work.
- **`dfb_mask_h_w_obj` is constructed on a CB that may not exist.**
  `moreh_bias_backward_single_core_hw.cpp:21-22` constructs the wrapper unconditionally, outside
  the runtime `if (do_mask_h || do_mask_w)`, while `SingleCoreProgramFactory` allocates `c_2` only
  when masking applies. Benign on Gen1 (the constructor only computes an L1 interface address);
  it is exactly why the conditional-binding pattern is needed there.
- **Misleading kernel-side name baked into a named argument.** The multi-core factory passes
  `num_tiles = batch_num * Ht` (`:40`) as reader RTA slot 1, and both the reader
  (`reader_..._h.cpp:13`) and the compute kernel (`..._multi_core_h.cpp:12`) unpack it into a
  local named `batch_num`. The value is right (a column spans `batch_num * Ht` tiles); the name
  understates it. Named after the kernel-side local per the recipe, with a comment at the schema
  site stating what the value is. Renaming the kernel local is an ops-team cleanup, not port work.
- **`c_2` is allocated unconditionally in the multi-core factory** (two tiles of L1 on every core
  even with no masking), unlike the single-core factory which guards the same allocation. Preserved
  faithfully — unconditional DFB in multi-core, conditional in single-core.
- No unreferenced kernel files. No descriptor type outside the audit's scan. No
  `GlobalCircularBuffer`, no `GlobalSemaphore`, no borrowed-memory CB, no `address_offset`, no
  aliased CB, no op-owned tensors, no offset-folded base pointer.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (base concept), both factories.
  The op has no `override_runtime_arguments`, so the framework refreshes tensor bindings on a cache
  hit and each factory writes exactly one `create_program_artifacts` method. No agreement problem —
  the audit's choice matches what the code shows.
- **Custom `compute_program_hash`**: none — default reflection-based hash. Nothing to leave alone.
- **Implementation notes**: the only device-operation-**class** edit the port forces is the two
  method signatures in the header (`ProgramDescriptor create_descriptor(...)` →
  `ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`) plus swapping
  `#include <tt-metalium/program_descriptors.hpp>` for `#include "ttnn/metal_v2_artifacts.hpp"`.
  None of the three sanctioned exceptions applies: no pybound `create_descriptor`
  (`moreh_linear_backward_nanobind.cpp:18-35` binds only the user-facing `ttnn::moreh_linear_backward`),
  no pybind-hook-only factory parameter, and the op already has a `program_factory_t` variant.

## Planned Spec Shape

Default is 1:1 with legacy. Resource names are declared **function-local** in each factory (as the
merged `moreh_mean` port does): the two factory `.cpp` files land in the same unity-build
translation unit, so no anonymous-namespace constants are introduced.

Shared names across the two factories, forced by the shared writer
(`writer_moreh_bias_backward.cpp` is compiled per-factory against that factory's schema, so its
binding vocabulary must match in both): DFB `out` (accessor `"out"`), tensor parameter `bias_grad`
(accessor `"dst"`), RTAs `num_tiles` / `start_id`. Names taken from the kernel's own vocabulary
(`cb_id_out` → `out`, `dst_addr` → `dst`), not from either factory's locals.

### Variant: `SingleCoreProgramFactory`

- **KernelSpecs** (3, one per legacy `KernelDescriptor`):
  - `reader` — `reader_moreh_bias_backward_hw.cpp`; `compile_time_args` **empty** (its only legacy
    CTAs were accessor plumbing); RTAs `num_tiles`, `start_id`, `mask_h`, `mask_w`, `do_mask_h`,
    `do_mask_w`; `hw_config = ttnn::create_reader_datamovement_config(device->arch())`;
    `opt_level` left at the Metal 2.0 default `O2` (matches legacy DM `O2`).
  - `writer` — `writer_moreh_bias_backward.cpp`; `compile_time_args` empty; RTAs `num_tiles`,
    `start_id`; `hw_config = ttnn::create_writer_datamovement_config(device->arch())`; `O2`.
  - `compute` — `moreh_bias_backward_single_core_hw.cpp`; `compile_time_args` empty (legacy was
    `{}`); RTAs `batch_num`, `Ht`, `Wt`, `do_mask_h`, `do_mask_w`; defines as legacy, **plus**
    `DO_MASK_H_W=1` when the mask DFB is bound; `hw_config =
    ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config)` (Style A);
    `compiler_options.opt_level = KernelBuildOptLevel::O3` **set explicitly**.
- **DataflowBufferSpecs** (6, or 5 when no masking — one per legacy `CBDescriptor`; no aliasing, no
  borrowed memory, so no `advanced_options`):
  | unique_id | entry_size | num_entries | data_format_metadata | legacy CB |
  |---|---|---|---|---|
  | `in0` | `tile_size(cb_data_format)` | 2 | `cb_data_format` | `c_0` |
  | `scaler` | `tile_size(cb_data_format)` | 1 | `cb_data_format` | `c_1` |
  | `mask_h_w` | `tile_size(cb_data_format)` | 2 | `cb_data_format` | `c_2` — **declared only when `do_mask_h \|\| do_mask_w`** |
  | `out` | `tile_size(cb_data_format)` | 1 | `cb_data_format` | `c_16` |
  | `intermed0` | `tile_size(cb_data_format)` | 1 | `cb_data_format` | `c_24` |
  | `intermed1` | `tile_size(fp32_dest_acc_en_data_format)` | 1 | `fp32_dest_acc_en_data_format` | `c_25` |

  `tile_format_metadata` left unset on all six — no legacy `format_descriptors[i].tile` was set.
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `output_grad` (spec from the input `MeshTensor`), `bias_grad` (spec from
  the output `MeshTensor`). `bias` gets none — it is host-only.
- **WorkUnitSpecs** (1): `{reader, writer, compute}` on `{0,0}`.
- **Op-owned tensors**: none.

DFB endpoint assignment, re-derived from the per-node kernel-touch census (not transcribed):

| DFB | reader | compute | writer | touchers | disposition |
|---|---|---|---|---|---|
| `in0` | `reserve_back`/`push_back` (`reader_..._hw.cpp:47,50`) | `wait_front`/`pop_front` (`..._single_core_hw.cpp:49,87`) + `reduce<… cb_in0 …>` input | — | 2 | **1P + 1C** |
| `scaler` | `fill_cb_with_value` → `reserve_back`/`push_back` (`moreh_common.hpp:99,108`) | `wait_front` (`:31`, never popped — fill-once/read-many) | — | 2 | **1P + 1C** |
| `mask_h_w` | `generate_mask_h_w` → `reserve_back(2)`/`push_back` (`moreh_common.hpp:273`) | `wait_front` + `copy_tile` (`:34,62,71`) | — | 2 | **1P + 1C**, both conditional |
| `out` | — | `reduce<… cb_out0>` output | `wait_front`/`pop_front` (`writer_...cpp:28,31`) | 2 | **1P + 1C** |
| `intermed0` | — | `reserve_back`/`push_back` (`:79,84`) **and** `reduce<… cb_intermed0 …>` input | — | **1** | **compute self-loop** (PRODUCER + CONSUMER) |
| `intermed1` | — | `reduce<… cb_intermed1>` output **and** `Accumulate::at(cb_intermed1, …)` reload (`:92,98,106`) | — | **1** | **compute self-loop** |

No DFB has ≥3 distinct touchers and none has two kernels locked to the same FIFO role, so
`allow_instance_multi_binding` is **not** set anywhere. No dead CB (every CB has ≥1 endpoint).

### Variant: `MultiCoreProgramFactory`

- **KernelSpecs** (4, or 3 without core group 2):
  - `reader` — `reader_moreh_bias_backward_h.cpp`; `compile_time_args` empty; RTAs `batch_num`,
    `Wt`, `Wt_per_core`, `start_id`, `mask_h`, `mask_w`, `do_mask_h`, `do_mask_w`;
    reader DM config; `O2`.
  - `writer` — same source and same vocabulary as the single-core factory's writer; RTAs
    `num_tiles`, `start_id`; writer DM config; `O2`.
  - `compute_g1` / `compute_g2` — `moreh_bias_backward_multi_core_h.cpp`, **two KernelSpecs of the
    same source**, differing only in the named CTA `units_per_core`
    (`num_cols_per_core_group_1` / `_2`); RTAs `batch_num`, `Ht`, `Wt_per_core`, `do_mask_h`,
    `do_mask_w`; defines as legacy; Style-A compute `hw_config`; `opt_level = O3` **on each**.
    `compute_g2` is built only when `has_core_group_2`.
- **DataflowBufferSpecs** (6): identical to the single-core table, except `mask_h_w` is declared
  **unconditionally** with `num_entries = 2`.
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `output_grad`, `bias_grad`.
- **WorkUnitSpecs** (2, or 1): `wu_g1 = {reader, writer, compute_g1}` on `core_group_1`;
  `wu_g2 = {reader, writer, compute_g2}` on `core_group_2` when present. Reader and writer belong
  to both work units, so their derived node set is the union — the legacy `all_cores`.
- **Op-owned tensors**: none.

Endpoint census is identical to the single-core table (same six roles, same touchers), reading the
multi-core kernels: reader `reader_..._h.cpp:45,48` / scaler at `:27-28` via
`calculate_and_prepare_reduce_scaler`, mask at `:31-32`; compute `..._multi_core_h.cpp:32,35,50,88`
and self-loops at `:80,85,93,99,107`. `mask_h_w` is a 1P+1C here too, unconditionally bound.
Each node hosts exactly **one** compute instance (the two groups are disjoint), so every shared-DFB
binding is an ordinary single-role binding — **not** the two-toucher 1P+1C case and **not**
`allow_instance_multi_binding`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_1` (`core_group_1`, `..._multi_core_program_factory.cpp:166-179`) and `compute_desc_2` (`core_group_2`, `:181-197`), both of `moreh_bias_backward_multi_core_h.cpp` | `compute_g1`, `compute_g2` — same source, differing only in the named CTA `units_per_core` | `wu_g1` (`core_group_1`), `wu_g2` (`core_group_2`) | `in0` CONSUMER · `scaler` CONSUMER · `mask_h_w` CONSUMER · `out` PRODUCER · `intermed0` PRODUCER+CONSUMER · `intermed1` PRODUCER+CONSUMER — each on **disjoint** node sets, so each is a legal single-role binding; no flag |

`SingleCoreProgramFactory`: none — no work-split multiplicity in legacy.

The per-group CTA is **dead** (the kernel reads no compile-time argument at all), which makes this
shape easy to mis-simplify into one `KernelSpec`. It is not simplified: collapsing it would drop the
multiplicity, and dropping the CTA is an owner decision.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._single_core_program_factory.cpp:128-129` | `TensorAccessorArgs(*output_grad.buffer()).get_compile_time_args()` → reader `compile_time_args` | `TensorParameter output_grad` + `TensorBinding{output_grad, "src"}` on the reader |
| `..._single_core_program_factory.cpp:130-131` | `TensorAccessorArgs(*bias_grad.buffer()).get_compile_time_args()` → writer `compile_time_args` | `TensorParameter bias_grad` + `TensorBinding{bias_grad, "dst"}` on the writer |
| `..._single_core_program_factory.cpp:182` | reader RTA slot 0 = `output_grad.buffer()` (the `Buffer*`-binding form — **not** `->address()`) | `TensorBinding` (address auto-injected per enqueue) |
| `..._single_core_program_factory.cpp:189` | writer RTA slot 0 = `bias_grad.buffer()` | `TensorBinding` |
| `..._multi_core_program_factory.cpp:132-133` | reader `TensorAccessorArgs` CTAs | `TensorBinding{output_grad, "src0"}` |
| `..._multi_core_program_factory.cpp:134-135` | writer `TensorAccessorArgs` CTAs | `TensorBinding{bias_grad, "dst"}` |
| `..._multi_core_program_factory.cpp:219` | reader RTA slot 0 = `output_grad_buf` | `TensorBinding` |
| `..._multi_core_program_factory.cpp:229` | writer RTA slot 0 = `bias_grad_buf` | `TensorBinding` |
| `reader_moreh_bias_backward_hw.cpp:20` / `reader_moreh_bias_backward_h.cpp:22` / `writer_moreh_bias_backward.cpp:16` | `constexpr auto …_args = TensorAccessorArgs<0>()` | dropped; `TensorAccessor(tensor::<name>)` |
| `reader_..._hw.cpp:12` / `reader_..._h.cpp:12` / `writer_...cpp:12` | `src_addr` / `src0_addr` / `dst_addr` = `arg_fetcher.get_next_arg_val<uint32_t>()` | dropped with the binding; the remaining RTAs become named, so no re-indexing is needed |
| `reader_..._hw.cpp:22-24`, `reader_..._h.cpp:23-25`, `writer_...cpp:17`, `..._single_core_hw.cpp:17-26`, `..._multi_core_h.cpp:18-27` | hardcoded CB-index constants (`constexpr uint32_t cb_id_in0 = 0`, `constexpr auto cb_in0 = tt::CBIndex::c_0`, …) | `DFBBinding` → `dfb::<name>` |
| every kernel, `ArgFetcher arg_fetcher; … get_next_arg_val<uint32_t>()` (`moreh_common.hpp:44` / `:128`) | positional RTA walk | `get_arg(args::<name>)` — a fixed run of distinct fields read once each, so **named**, not varargs. `ArgFetcher` disappears from the ported kernels; the donor header is untouched |

**Page-size 3rd-argument CTAs/RTAs**: none — no accessor in the op passes a third constructor
argument, so there is nothing to drop.
**Semaphore-ID RTAs**: none — the op has no semaphores.
**Magic CB indices in CTAs**: none on the host side — the kernels hardcoded their CB indices as
source literals rather than reading them from CTAs, so no host CTA slot carried one. (The literals
themselves are removed by the `DFBBinding` swap, listed above.)
**Positional CTAs**: the only non-accessor positional CTA in the op is the multi-core compute
`{num_cols_per_core_group_N}` → named `units_per_core`.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  `intermed0` and `intermed1` on the compute `KernelSpec` (both PRODUCER and CONSUMER, shared
  accessor name), in **both** factories. The legitimate accumulator/staging case — compute-side, so
  supported on Gen2 as well as Gen1; no Quasar debt.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings):
  `mask_h_w` in `SingleCoreProgramFactory` only — the `DataflowBufferSpec`, the reader's PRODUCER
  binding and the compute's CONSUMER binding are all gated on `do_mask_h || do_mask_w`, with a
  matching `DO_MASK_H_W=1` define emitted to **both** kernels and `#ifdef`-gated kernel-side
  references (the `DataflowBuffer` construction, the `wait_front`, and both `copy_tile` sub-blocks).
  The existing runtime `if`s stay nested inside the `#ifdef`s — redundant when compiled in, and
  removing them would be kernel-logic surgery.
- [Anti-pattern avoided: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta):
  `MultiCoreProgramFactory` keeps two compute `KernelSpec`s in two `WorkUnitSpec`s rather than one
  spec with the per-group value demoted to an RTA.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `dfb::name` passed straight to `compute_kernel_hw_startup`, `copy_tile`,
  `copy_tile_to_dst_init_short`, `reconfig_data_format_srca`, `pack_reconfig_data_format`,
  `pack_tile`, `compute_kernel_lib::reduce<>` (NTTP position),
  `compute_kernel_lib::Accumulate::at`, and
  `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<>` (NTTP position). No `.id`
  extraction, no temporary wrappers.
- [Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  intra-op `writer_moreh_bias_backward.cpp` — **rung 0**, converted in place because both binding
  factories convert in this change. No fork, no pointer comment, nothing left behind.
- **CB→DFB whitelist §A, member-getter form**: `get_tile_size(cb_id)` → `dfb.get_tile_size()` at
  `reader_..._hw.cpp:43`, `reader_..._h.cpp:39`, `writer_...cpp:24`. All three legacy declarations
  are `const auto`, **not** `constexpr`, so the member getter is correct and the
  `get_tile_size(dfb::name)` token form is not used anywhere in this port (no Gen1-only token-form
  debt to record).

## Hardware configuration and compiler options

- **DM kernels**: both resolved triples are exactly the role defaults
  (`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`, no custom `processor` / `noc` /
  `noc_mode` anywhere in the op), and the roles match the names. →
  `ttnn::create_reader_datamovement_config(device->arch())` /
  `ttnn::create_writer_datamovement_config(device->arch())`. No `DM_DYNAMIC_NOC`, so the
  paired-per-node concern does not arise.
- **Compute kernels — Style A, no dropped field.** Both factories resolve a TTNN
  `DeviceComputeKernelConfig` (built by `init_device_compute_kernel_config` in
  `ttnn::prim::moreh_bias_add_backward`, `device/moreh_linear_backward_device_operation.cpp:79`)
  and destructure it with `get_compute_kernel_config_args`, which is a pure destructure. The
  factories set `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`, `unpack_to_dest_mode` and
  `math_approx_mode` — **every** resolved field with a Metal 2.0 counterpart; `packer_l1_acc` has
  none (no action). So `ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config)`
  carries the values across, including the two non-1:1 transforms
  (`math_approx_mode` bool → `Precision`, and `dst_full_sync_en` → `double_buffer_dest`
  **inversion**), which are **not** re-applied by hand on top of the helper.
  `bfp8_pack_precise` is never set, so `bfp_pack_precision_mode` stays at its matching default.
- **`opt_level`**: `O3` set explicitly on **every** compute `KernelSpec` (single-core `compute`,
  multi-core `compute_g1` **and** `compute_g2`) — legacy `ComputeConfigDescriptor` resolves to `O3`
  while Metal 2.0's `CompilerOptions` defaults to `O2`. The four DM specs need nothing.
- **`unpack_modes`** — the two factories deliberately differ. See below.

### `unpack_modes` — the highest-risk item, and a broader entry set than the brief specified

Legacy builds `std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, Default)`
in both factories, and only `MultiCoreProgramFactory` modifies it:
`unpack_to_dest_mode[CBIndex::c_25] = UnpackToDestFp32` when `fp32_dest_acc_en` (`:159-163`).
Metal 2.0 keys the same information by DFB name and requires an explicit entry wherever a compute
kernel **consumes** a DFB whose `data_format_metadata` is `Float32` while `enable_32_bit_dest` is
set (`tt_metal/impl/metal2_host_api/program_spec.cpp:1049-1077`).

| Factory | legacy | Metal 2.0 | why |
|---|---|---|---|
| `MultiCoreProgramFactory` | `unpack_to_dest_mode[c_25] = UnpackToDestFp32` when `fp32_dest_acc_en` | `{intermed1, UnpackMode::UnpackToDest}` when `fp32_dest_acc_en` | faithful translation of the one explicit legacy setting |
| `SingleCoreProgramFactory` | never modified — every entry `Default` | `{intermed1, UnpackMode::UnpackToSrc}` | legacy `Default` ⇒ `UnpackToSrc`; Metal 2.0 requires the entry explicitly under fp32 |
| **both** | the remaining entries, all `Default` | `{in0, …}`, `{scaler, …}`, `{mask_h_w, …}`, `{intermed0, …}` all `UnpackMode::UnpackToSrc` | see below |

**Why the last row, which the brief did not call for.** The brief reasoned that `intermed0`
(and the other compute-consumed DFBs) "never" carry `Float32` because their format is
`cb_data_format`. That holds only while `output_grad.dtype() != DataType::FLOAT32` — and nothing
gates the dtype: neither `MorehBiasAddBackwardOperation::validate_inputs`
(`device/moreh_linear_backward_device_operation.cpp:17-26`) nor
`ttnn::moreh_linear_backward` (`moreh_linear_backward.cpp:106-170`) restricts it, and
`ttnn::prim::moreh_bias_add_backward` is reachable directly. With a `Float32` `output_grad` **and**
`fp32_dest_acc_en`, every compute-consumed DFB is a `Float32` DFB, so the validator would require
an entry for each — and a spec carrying only the `intermed1` entry would `TT_FATAL` where legacy
ran. Emitting the whole legacy `Default` row avoids that behavior change at zero cost:
`UnpackToSrc` is **always** legal (`program_spec.cpp:1004-1005`) and
`BuildUnpackToDestModeVector` maps it back to `UnpackToDestMode::Default`
(`program_spec.cpp:2711-2714`), so the internal vector the JIT sees is byte-identical to legacy's
in **every** configuration. This transcribes the legacy vector rather than adding a value to it.

The `mask_h_w` entry is gated on the same condition as its binding in the single-core factory — the
validator rejects an entry naming a DFB the kernel does not bind.

Applied via `std::get<ComputeGen1Config>(compute_hw).unpack_modes = …` on the value returned by
`to_compute_hardware_config`, per the recipe. The `unpack_modes(cfg)` common-field accessor is
deliberately **not** used — converting to it belongs to the `gen2_hardware_configs` post-port pass.

## Deferred / Flagged

- **New finding during planning**: the `unpack_modes` entry set is broader than the brief
  specified, for the `Float32`-`output_grad` case described above. Recorded in
  `METAL2_PORT_REPORT.md` under Friction (the brief's premise) and Open items (the ungated dtype).
- Everything else planning turned up was already in the audit's Misc anomalies (dead CTA,
  `batch_num` misnomer, unconditional multi-core `c_2`, unconditional `dfb_mask_h_w_obj`
  construction) and is preserved verbatim; each is recorded under Flags above and in the report.
- No structural issue the audit missed. No stop signal.
