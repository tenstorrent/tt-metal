# Port Plan — `ttnn/cpp/ttnn/operations/moreh/moreh_norm`

Port plan for `moreh_norm`, ported from the legacy `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this plan:** all three factories of `MorehNormOperation` — `ProgramFactoryWOther`,
`ProgramFactoryHOther`, `ProgramFactoryNCOther`. They convert together (see *TTNN ProgramFactory*).

> **Live vs dead kernel tree.** Only paths containing `ord_other/` are live. The three files under
> `device/moreh_norm_{h,w,other}/kernels/` are unreferenced, two of them share a basename with a live
> kernel, and they are **out of scope** — this port does not read or edit them. Every path in this
> plan carries the `ord_other/` segment; that is the check.

---

## Legacy Inventory

### Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** — each factory exposes
  `static tt::tt_metal::ProgramDescriptor create_descriptor(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`
  (`device/moreh_norm_device_operation.hpp:34, 41, 48`), building and returning a `ProgramDescriptor`
  in place.
- Variants: **three factories**, nested structs inside `MorehNormOperation`, named in the
  `program_factory_t` variant at `device/moreh_norm_device_operation.hpp:54`. Dispatch is by reduced-dim
  position only (`device/moreh_norm_device_operation.cpp:43-54`): `dim == rank-1` → W, `dim == rank-2` → H,
  otherwise NC. Interleaved only; no sharded path.
- Custom `compute_program_hash`: **none** — the op already uses the default reflection-based hash.
  Nothing to delete.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's
TTNN factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

**Attribute space actually reachable.** The device op is only ever entered for `p ∈ {0, +INF, -INF}`;
the host wrapper (`moreh_norm.cpp:29-60`) routes every other `p` through `moreh_abs_pow` + `moreh_sum`.
That is why each factory's `IS_ZERO` / `MINUS_INF` / `REDUCE_OP` define matrix is exhaustive. The port
reproduces the define matrix verbatim and adds no new branch.

**Shared structure across the three factories.** All three are the same pipeline — reader fills a
`one` tile (+ a mask tile for W/H) and streams input tiles → compute applies `f(x)`, accumulates along
the reduced dim, reduces → writer drains. NC has no mask CB and no `reduce<>` call (its accumulator
drains straight to the output). The per-resource tables below are therefore given per variant.

---

### Variant: W — `ProgramFactoryWOther`

Source: `device/ord_other/moreh_norm_program_factory_w_other.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/ord_other/moreh_norm_w/kernels/reader_moreh_norm_w.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` only (`:159`) | none | per core (`:262-269`): `input_buf` (`Buffer*`), `is_dram(input)`, `num_units_per_core`, `Wt`, `tile_offset`, `origin_w` | none | none | `ReaderConfigDescriptor{}` (`:168`) |
| writer | `device/ord_other/moreh_norm_w/kernels/writer_moreh_norm_w.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` only (`:161`) | none | per core (`:272-278`): `output_buf` (`Buffer*`), `is_dram(output)`, `num_units_per_core`, `Wt`, `tile_offset` | none | none | `WriterConfigDescriptor{}` (`:175`) |
| compute_1 | `device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` | `core_group_1` | `{}` (`:204`) | none | per core (`:242-248`): `num_units_per_core`, `Wt`, `origin_w` | none | `compute_defines` (`:205`) | `ComputeConfigDescriptor{...}` (`:206-212`) |
| compute_2 | same source | `core_group_2` (only when non-empty, `:215-229`) | `{}` (`:220`) | none | per core (`:251-257`): same three | none | `compute_defines` (`:221`) | same `ComputeConfigDescriptor` (`:222-228`) |

`compute_defines` (`:180-192`): `REDUCE_DIM = "ReduceDim::REDUCE_ROW"` always; then
`p == 0` → `REDUCE_OP = "PoolType::SUM"`, `IS_ZERO = "1"`; else `REDUCE_OP = "PoolType::MAX"` and,
when `p == -inf`, `MINUS_INF = "1"`.

`ComputeConfigDescriptor` fields set: `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`,
`unpack_to_dest_mode` (a `NUM_CIRCULAR_BUFFERS`-long vector, **every entry `UnpackToDestMode::Default`**,
`:198`), `math_approx_mode`. All five come from
`get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config)` (`:55-56`).
`packer_l1_acc` is destructured but never used. `bfp8_pack_precise` is left at its default.

#### CBs

`total_size` is `n_t * tile_size(fmt)` with every `n_t == 1`, so each CB is exactly one tile.
`cb_data_format = datatype_to_dataformat_converter(input.dtype())`;
`intermed_data_format = fp32_dest_acc_en ? Float32 : cb_data_format` (`:69-70`).
No `tile` field set on any `CBFormatDescriptor` → no non-default tile geometry.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` input (`:84-92`) | `1 * tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | — |
| `c_1` one (`:93-101`) | `1 * tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | — |
| `c_2` mask_w (`:102-110`) | `1 * tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | — |
| `c_16` output (`:111-119`) | `1 * tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | — |
| `c_24` val = `f(x)` (`:120-128`) | `1 * tile_size(intermed_data_format)` | `all_cores` | `intermed_data_format` | `tile_size(intermed_data_format)` | — |
| `c_25` cal = accumulator (`:129-137`) | `1 * tile_size(intermed_data_format)` | `all_cores` | `intermed_data_format` | `tile_size(intermed_data_format)` | — |
| `c_26` reduce result (`:138-146`) | `1 * tile_size(intermed_data_format)` | `all_cores` | `intermed_data_format` | `tile_size(intermed_data_format)` | — |

No `GlobalCircularBuffer` anywhere (no `.global_circular_buffer`, no `remote_cb_*`, no
`set_globally_allocated_address`). No Buffer-backed / borrowed-memory CB. No aliased CB (every
`format_descriptors` list has exactly one element).

#### Semaphores

none — the op declares no semaphores of any kind.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._w_other.cpp:159` (`TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args)`) | `tensor_args.input` | reader RTA 0 (`:264`, typed `Buffer*`) → `reader_moreh_norm_w.cpp:12` `input_addr`, `:24` `input_args`, `:25` construction |
| `..._w_other.cpp:161` (`TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args)`) | `output` (`tensor_return_value_t&`) | writer RTA 0 (`:274`, typed `Buffer*`) → `writer_moreh_norm_w.cpp:14` `output_addr`, `:23` `output_args`, `:24` construction |

Both are **Case 1** (consumed only through a `TensorAccessor`; no raw base-pointer arithmetic). All
constructions are the 2-argument form — no page-size third argument anywhere in the op.

#### Work split

- Driver: `tt::tt_metal::split_work_to_cores(grid, num_units)` (`:58-64`), with
  `num_units = input.physical_volume() / H / W * Ht` (`:44`).
- `(num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2)`.
- Core iteration order is `CoreCoord{i / num_cores_y, i % num_cores_y}` (`:237`), and `tile_offset`
  accumulates `num_units_per_core * Wt` per core (`:280`).

---

### Variant: H — `ProgramFactoryHOther`

Source: `device/ord_other/moreh_norm_program_factory_h_other.cpp`. **Structurally identical to W**;
only the deltas are listed.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/ord_other/moreh_norm_h/kernels/reader_moreh_norm_h.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` (`:152`) | `:244-252`: `input_buf`, `is_dram(input)`, `num_cols_per_core`, `tile_offset`, `Ht`, `Wt`, `origin_h` | none | `ReaderConfigDescriptor{}` (`:161`) |
| writer | `device/ord_other/moreh_norm_h/kernels/writer_moreh_norm_h.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` (`:154`) | `:255-256`: `output_buf`, `is_dram(output)`, `num_cols_per_core`, `tile_offset` | none | `WriterConfigDescriptor{}` (`:168`) |
| compute_1 | `device/ord_other/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` | `core_group_1` | `{}` (`:196`) | `:234-235`: `num_cols_per_core`, `Ht`, `origin_h` | `compute_defines` | `ComputeConfigDescriptor{...}` (`:198-204`) |
| compute_2 | same source | `core_group_2` when non-empty (`:207-221`) | `{}` (`:212`) | `:238-239`: same three | `compute_defines` | same (`:214-220`) |

`compute_defines` differs from W in one entry only: `REDUCE_DIM = "ReduceDim::REDUCE_COL"` (`:174`).

#### CBs

Same seven CBs, same sizes and formats (`:77-139`); `c_2` is `mask_h` instead of `mask_w`.

#### Semaphores

none.

#### Tensor accessors

`..._h_other.cpp:152` / `:246` → `reader_moreh_norm_h.cpp:12, 25, 26` (input, Case 1);
`..._h_other.cpp:154` / `:256` → `writer_moreh_norm_h.cpp:14, 22, 23` (output, Case 1).

#### Work split

`split_work_to_cores(grid, num_units)` (`:51-57`) with `num_units = input.physical_volume() / H / W * Wt`
(`:37`). `tile_offset` accumulates `num_cols_per_core` per core (`:258`).

---

### Variant: NC — `ProgramFactoryNCOther`

Source: `device/ord_other/moreh_norm_program_factory_nc_other.cpp`. Same pipeline, **no mask CB and
no `reduce<>`**; the accumulator drains straight to the output.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/ord_other/moreh_norm_nc/kernels/reader_moreh_norm_nc.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` (`:139`) | `:236-244`: `input_buf`, `is_dram(input)`, `num_output_tiles_per_core`, `tile_offset`, `outer_stride`, `num_inner_tiles`, `num_reduced_tiles_along_dim` | none | `ReaderConfigDescriptor{}` (`:148`) |
| writer | `device/ord_other/moreh_norm_nc/kernels/writer_moreh_norm_nc.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` (`:141`) | `:247-248`: `output_buf`, `is_dram(output)`, `num_output_tiles_per_core`, `tile_offset` | none | `WriterConfigDescriptor{}` (`:155`) |
| compute_1 | `device/ord_other/moreh_norm_nc/kernels/moreh_norm_nc_kernel.cpp` | `core_group_1` | `{}` (`:180`) | `:218-223`: `num_output_tiles_per_core`, `num_reduced_tiles_along_dim` | `compute_defines` | `ComputeConfigDescriptor{...}` (`:182-188`) |
| compute_2 | same source | `core_group_2` when non-empty (`:191-205`) | `{}` (`:196`) | `:226-231`: same two | `compute_defines` | same (`:198-204`) |

`compute_defines` (`:160-168`): **no `REDUCE_DIM`, no `REDUCE_OP`** (there is no `reduce<>` call);
`p == 0` → `IS_ZERO = "1"`, else `p == -inf` → `MINUS_INF = "1"`.

#### CBs

Five CBs (`:82-126`): `c_0` input, `c_1` one, `c_16` output (all `cb_data_format`), `c_24` val,
`c_25` cal (both `intermed_data_format`). One tile each. No mask CB, no `c_26`.

#### Semaphores

none.

#### Tensor accessors

`..._nc_other.cpp:139` / `:238` → `reader_moreh_norm_nc.cpp:12, 24, 25` (input, Case 1);
`..._nc_other.cpp:141` / `:248` → `writer_moreh_norm_nc.cpp:14, 22, 23` (output, Case 1).

#### Work split

`split_work_to_cores(grid, num_output_tiles)` (`:58-64`) with
`num_output_tiles = output.physical_volume() / TILE_HW` (`:34`). `tile_offset` accumulates
`num_output_tiles_per_core` per core (`:250`).

---

### Shared kernels

**none.** All nine live `kernel_source` paths are under `device/ord_other/…` and are op-exclusive:

- No kernel is bound by more than one of this op's own factories — each factory owns its own copy of
  reader / compute / writer (verified against the three factories' `kernel_source` strings).
- `grep -rl <basename> ttnn/cpp/ttnn/operations/` for each of the nine basenames returns only this op
  (plus the family `CMakeLists.txt` install glob, which is not a consumer, and — for
  `moreh_norm_h_kernel.cpp` / `moreh_norm_w_kernel.cpp` — the *dead* sibling under
  `device/moreh_norm_{h,w}/kernels/`, which no factory binds).
- No `*_metal2.*` fork exists anywhere beside any of them (checked locationally by listing each
  kernel directory).

So there is nothing to co-port, no fork to create, and no pointer comment to leave. All nine kernels
are converted in place.

The **donor headers** the kernels call into are out-of-op but are not shared-kernel cases — they are
kernel-lib / shared-pool callees that the boundary features bridge:

| Donor | Used by | Crossing mechanism |
|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` — `fill_cb_with_value`, `generate_mask_w`, `generate_mask_h`, `Scalar` | all three readers | takes `DataflowBuffer` **by value** → pass the local `DataflowBuffer(dfb::name)` object; no donor change |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` — `compute_kernel_lib::reduce<…>` | W, H compute | `uint32_t` CB-id **NTTPs** → `DFBAccessor`'s `constexpr operator uint32_t()` converts in template-argument position |
| `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` — `copy_tile_init_with_dt`, `pack_tile_with_dt`, `add_tiles_init_with_dt`, and the compute LLK re-exports | all three compute kernels | takes `DataflowBuffer` **by value** → pass the local object |
| `tt_metal/hw/inc/api/**` (`dataflow/noc.h`, `dataflow/dataflow_buffer.h`, `dataflow/dataflow_api.h`, `tensor/noc_traits.h`) | all six dataflow kernels | LLK / HAL; unchanged |

No call site outside the op directory requires a `sem::` or `tensor::` handle, so the recipe's
boundary-rule assumption holds.

### Flags

1. **Three unreferenced kernel files, two basename-colliding with live ones** —
   `device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp`,
   `device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`,
   `device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp`. No factory, op, or test binds them;
   only the family install glob at `ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:44-46` still copies
   them. **Not audited, not touched by this port.** (Audit Misc anomaly 1; deletion is the ops team's
   call — routed to the port report.)
2. **`get_floored_p_and_decimal_and_p_is_negative`** (`device/moreh_norm_device_operation.cpp:14-22`,
   declared `…hpp:14`) is dead in this op. Left alone (device-op-class code is off-limits); routed to
   the report.
3. **Six dead `*_is_dram` RTAs** — every dataflow kernel reads a DRAM flag it never uses
   (`reader_moreh_norm_{w,h,nc}.cpp:13`, `writer_moreh_norm_{w,h,nc}.cpp:15`). **Preserved** by the
   port (named, still passed by the host); removal is the ops team's call.
4. **NC's `one` tile is produced and consumed but never read** (`reader_moreh_norm_nc.cpp:27-30` fills
   `c_1`; `moreh_norm_nc_kernel.cpp:37, 137` waits/pops it; no compute op reads it — the NC path has
   no `reduce<>` so it needs no scaler). The CB is genuinely live in the endpoint census (1P+1C), so
   this is a waste finding, **not** a dead-CB drop. Preserved verbatim.
5. **Deprecated `tt::CB` enum in the compute kernels** (`tt::CB::c_in0` / `c_out0` / `c_intermed0`)
   while the factories use `tt::CBIndex::c_N`. Both resolve to the same values. These lines are
   *removed by the port anyway* — they are exactly the magic-CB-index constants that become
   `dfb::` handles — so the inconsistency disappears as a side effect, not as a bundled cleanup.
6. **Runtime-counter CB-id idiom, in three spellings.** All six dataflow kernels open with
   `uint32_t cb_id{0};` / `{16};` then `const auto cb_id_x = cb_id++;`; the **NC compute** kernel uses
   a mutable `std::uint8_t input_id{tt::CB::c_in0}; const auto cb_x = input_id++;`
   (`moreh_norm_nc_kernel.cpp:12-29`); the **W and H compute** kernels use `constexpr auto`
   (`moreh_norm_w_kernel.cpp:14-35`, `moreh_norm_h_kernel.cpp:14-35`). All three spellings collapse to
   the same thing post-port — the counter and the derived ids are *deleted*, and each
   `DataflowBuffer` is constructed straight from its `dfb::` token. Nothing needs the counter to
   survive, so the constexpr-vs-runtime distinction the audit flagged does not actually force
   different treatment.
7. **No unreferenced kernel file inside the live tree** — all nine `ord_other/` kernels are bound.

---

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: **none** — the op already uses the default reflection-based hash.
  Nothing to delete.
- **Implementation notes**:
  - All three factories convert in the same change, so `program_factory_t`
    (`device/moreh_norm_device_operation.hpp:54`) holds no mixed-concept transition state. Each
    factory's `create_descriptor` declaration is **replaced** by
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`.
    The replacement is mandatory, not optional: `ProgramSpecFactoryConcept` is defined as
    `requires { &T::create_program_artifacts; } && … && !ProgramDescriptorFactoryConcept<T>`
    (`ttnn/api/ttnn/operation_concepts.hpp:119-121`), so leaving `create_descriptor` in place would
    make the factory satisfy two concepts and trip the `AllFactoriesValid` `static_assert`.
  - **No pybind removal.** `moreh_norm_nanobind.cpp:38-49` binds only the user-facing
    `ttnn::moreh_norm` free function — no `create_descriptor` exposure, no factory internals. Nothing
    in the pybind layer changes.
  - **No other device-op-class edit.** `validate_inputs`, `select_program_factory`,
    `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` are untouched.
  - **Op-owned tensors**: none. `ProgramArtifacts::op_owned_tensors` is left defaulted.
  - **Optional output**: `tensor_args_t` carries a `const std::optional<Tensor>& output`, but the
    factory only ever sees the concrete `tensor_return_value_t& output` that `create_output_tensors`
    returned (`device/moreh_norm_device_operation.cpp:93-101` returns a caller-supplied tensor
    verbatim, else a freshly-allocated one). Either way that is **one** `TensorParameter` for the
    output — the optionality is resolved before the factory runs and needs no conditional binding.
  - **MeshTensor extraction**: each factory extracts `const auto& input = tensor_args.input.mesh_tensor();`
    and `const auto& out = output.mesh_tensor();` once at entry and works with those throughout
    (shape / dtype / device queries, `tensor_spec()`, `TensorArgument`). The one place the
    `ttnn::Tensor` is still needed is the moreh helper `is_dram(const Tensor&)`
    (`moreh_helper_functions.hpp:19`), which keeps the dead-but-preserved `*_is_dram` RTA byte-identical;
    those two call sites read `is_dram(tensor_args.input)` / `is_dram(output)`.
  - **Spec-name constants are function-local.** `ttnn_op_moreh` is a unity build
    (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:7`, `TT_ENABLE_UNITY_BUILD`), which merges the three
    factory `.cpp`s into one translation unit and merges their anonymous namespaces with them. Rather
    than prefix every constant per factory, each factory declares its typed name constants
    (`KernelSpecName` / `DFBSpecName` / `TensorParamName`) as `const` locals at the top of
    `create_program_artifacts`. Function-local names cannot collide across TUs, so the
    [unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
    hazard is avoided without introducing a shared header or per-factory prefixes. All uses are inside
    that one function.

---

## Planned Spec Shape

Default is 1:1 with legacy, and it holds throughout: no DFB is dropped, added, merged, or split.

### Variant: W

- **KernelSpecs** (4, or 3 when `core_group_2` is empty) — one per legacy `KernelDescriptor`:
  | unique_id | source | hw_config | CTAs | RTA schema |
  |---|---|---|---|---|
  | `reader` | `…/reader_moreh_norm_w.cpp` | `create_reader_datamovement_config(arch)` | **none** | `input_is_dram`, `num_rows_per_core`, `Wt`, `tile_offset`, `origin_w` |
  | `writer` | `…/writer_moreh_norm_w.cpp` | `create_writer_datamovement_config(arch)` | **none** | `output_is_dram`, `num_rows_per_core`, `Wt`, `tile_offset` |
  | `compute_g1` | `…/moreh_norm_w_kernel.cpp` | `to_compute_hardware_config(arch, cfg)` + `unpack_modes` | **none** | `num_rows_per_core`, `Wt`, `origin_w` |
  | `compute_g2` | same source | same | **none** | same three |
- **DataflowBufferSpecs** (7) — one per legacy `CBDescriptor`; `entry_size = tile_size(fmt)`,
  `num_entries = 1`, `data_format_metadata = fmt`. No `tile_format_metadata` (legacy set no `tile`).
  No `borrowed_from`, no `alias_with`, no `allow_instance_multi_binding`.
  | DFB `unique_id` | legacy CB | format | reader | writer | compute |
  |---|---|---|---|---|---|
  | `input` | `c_0` | `cb_data_format` | PRODUCER `"input"` | — | CONSUMER `"x"` |
  | `one` | `c_1` | `cb_data_format` | PRODUCER `"one"` | — | CONSUMER `"one"` |
  | `mask_w` | `c_2` | `cb_data_format` | PRODUCER `"mask_w"` | — | CONSUMER `"mask_w"` |
  | `output` | `c_16` | `cb_data_format` | — | CONSUMER `"output"` | PRODUCER `"y"` |
  | `val` | `c_24` | `intermed_data_format` | — | — | **PRODUCER + CONSUMER** `"val"` (self-loop) |
  | `cal` | `c_25` | `intermed_data_format` | — | — | **PRODUCER + CONSUMER** `"cal"` (self-loop) |
  | `reduce` | `c_26` | `intermed_data_format` | — | — | **PRODUCER + CONSUMER** `"reduce"` (self-loop) |
- **SemaphoreSpecs**: none — legacy declares none.
- **TensorParameters** (2): `input` (`= input.tensor_spec()`, bound by `reader` as `tensor::input`),
  `output` (`= out.tensor_spec()`, bound by `writer` as `tensor::output`). Strict matching — no
  `advanced_options` relaxation (see *TensorParameter relaxation* below).
- **WorkUnitSpecs** (2, or 1 when `core_group_2` is empty):
  - `wu_g1`: `{reader, writer, compute_g1}` over `core_group_1`
  - `wu_g2`: `{reader, writer, compute_g2}` over `core_group_2`

  This reproduces legacy placement exactly — `reader`/`writer` belong to **both** work units, so their
  derived node set is `core_group_1 ∪ core_group_2 = all_cores`, while each compute spec stays on its
  own group. It is also what the local-DFB invariant requires: every DFB's producer and consumer are
  co-members of the same work unit(s), so the per-node census
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1308-1391`) sees exactly 1 producer + 1 consumer on
  every node. Declaring one work unit over `all_cores` instead would leave `input`/`output` with a
  producer but no consumer on group-2 nodes.

### Variant: H

Identical to W with three substitutions: kernel sources are the `moreh_norm_h/` trio; the `mask_w`
DFB becomes `mask_h` (accessor `"mask_h"`); and the per-kernel RTA names follow the H kernels'
own vocabulary — reader `input_is_dram, num_cols_per_core, tile_offset, Ht, Wt, origin_h`; writer
`output_is_dram, num_cols_per_core, tile_offset`; compute `num_cols_per_core, Ht, origin_h`.
Same 7 DFBs, same 2 TensorParameters, same 2 WorkUnitSpecs, same self-loop trio.

### Variant: NC

- **KernelSpecs** (4, or 3): `reader` / `writer` / `compute_g1` / `compute_g2` over the
  `moreh_norm_nc/` trio, no CTAs, same hw_config helpers. RTA schemas: reader
  `input_is_dram, num_output_tiles_per_core, tile_offset, outer_stride, num_inner_tiles, num_reduced_tiles_along_dim`;
  writer `output_is_dram, num_output_tiles_per_core, tile_offset`; compute
  `num_output_tiles_per_core, num_reduced_tiles_along_dim`.
- **DataflowBufferSpecs** (5): `input` (`c_0`), `one` (`c_1`), `output` (`c_16`) — plain 1:1;
  `val` (`c_24`), `cal` (`c_25`) — **self-loop** on compute. No `mask_*`, no `reduce`.
- **SemaphoreSpecs**: none. **TensorParameters** (2): `input`, `output`. **WorkUnitSpecs**: 2 (or 1),
  same shape as W.

### Op-owned tensors

none — the legacy factories allocate no device tensor beyond the op's declared io.

---

## Preserved Multiplicity

Each factory emits **two** compute `KernelDescriptor`s from **one** source over **disjoint** node sets.
This is the ordinary disjoint-node work split, *not* the same-grid dual-instance shape — each node
runs exactly one compute instance, so every DFB stays a plain per-node 1:1 (or a plain per-node
self-loop). Neither the `allow_instance_multi_binding` flag nor a 1P+1C reassignment is involved.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| W `compute_desc_1` (`..._w_other.cpp:200-212`), `compute_desc_2` (`:214-229`) of `moreh_norm_w_kernel.cpp` | `compute_g1`, `compute_g2` | `wu_g1` (`core_group_1`), `wu_g2` (`core_group_2`) | `input` CONSUMER · `one` CONSUMER · `mask_w` CONSUMER · `output` PRODUCER · `val` / `cal` / `reduce` PRODUCER **and** CONSUMER (self-loop) |
| H `compute_desc_1` (`..._h_other.cpp:192-204`), `compute_desc_2` (`:206-221`) of `moreh_norm_h_kernel.cpp` | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `input` C · `one` C · `mask_h` C · `output` P · `val` / `cal` / `reduce` P+C |
| NC `compute_desc_1` (`..._nc_other.cpp:176-188`), `compute_desc_2` (`:190-205`) of `moreh_norm_nc_kernel.cpp` | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `input` C · `one` C · `output` P · `val` / `cal` P+C |

**Not a CTA→RTA demotion risk.** Both legacy compute descriptors already carry
`compile_time_args = {}`; the per-group work count (`num_units_per_core_group_{1,2}`) travels as an
**RTA** in legacy and stays an RTA here. There is no per-group CTA to demote, so the
[demoting-per-group-CTA anti-pattern](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
cannot arise. The two specs are still kept separate — collapsing them into one would break the
`WorkUnitSpec` → node-set derivation that keeps each compute instance on its own group.

**Self-loop × multiplicity is explicitly legal.** Both compute specs self-loop `val` / `cal`
(/ `reduce`), so each of those DFBs has a two-element producer set and an identical two-element
consumer set. The validator permits exactly this — *"the producer set must equal the consumer set as
sets of `KernelSpec*` … This permits the natural pattern of multiple same-source KernelSpecs each
self-looping the DFB on their disjoint node ranges"*
(`tt_metal/impl/metal2_host_api/program_spec.cpp:1425-1444`).

---

## Dropped Plumbing

`TensorAccessorArgs` is the **only** compile-time arg in the entire op, and both compute descriptors
already declare `compile_time_args = {}`. So after the bindings are expressed, **every kernel in all
three factories has an empty `compile_time_args` table** — there are no positional CTAs left to name
and no `next_compile_time_args_offset()` chain anywhere to unwind.

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._w_other.cpp:159` | `TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args)` | `TensorParameter{input}` + `TensorBinding{INPUT, "input"}` on `reader` |
| `..._w_other.cpp:161` | `TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args)` | `TensorParameter{output}` + `TensorBinding{OUTPUT, "output"}` on `writer` |
| `..._w_other.cpp:264` (reader RTA slot 0) | `input_buf` (typed `Buffer*` into `emplace_runtime_args`) | `TensorBinding` — the base address is auto-injected per enqueue |
| `..._w_other.cpp:274` (writer RTA slot 0) | `output_buf` (typed `Buffer*`) | `TensorBinding` |
| `reader_moreh_norm_w.cpp:12` | `const auto input_addr = get_arg_val<uint32_t>(i++)` | deleted |
| `reader_moreh_norm_w.cpp:24-25` | `constexpr auto input_args = TensorAccessorArgs<0>();` + `TensorAccessor(input_args, input_addr)` | `TensorAccessor(tensor::input)` |
| `writer_moreh_norm_w.cpp:14` | `const auto output_addr = get_arg_val<uint32_t>(i++)` | deleted |
| `writer_moreh_norm_w.cpp:23-24` | `TensorAccessorArgs<0>()` + `TensorAccessor(output_args, output_addr)` | `TensorAccessor(tensor::output)` |
| `..._h_other.cpp:152` / `:154` / `:246` / `:256` | same four forms | same four replacements |
| `reader_moreh_norm_h.cpp:12, 25-26` · `writer_moreh_norm_h.cpp:14, 22-23` | same | same |
| `..._nc_other.cpp:139` / `:141` / `:238` / `:248` | same four forms | same four replacements |
| `reader_moreh_norm_nc.cpp:12, 24-25` · `writer_moreh_norm_nc.cpp:14, 22-23` | same | same |
| `reader_moreh_norm_w.cpp:19-22` · `reader_moreh_norm_h.cpp:20-23` · `reader_moreh_norm_nc.cpp:20-22` | `uint32_t cb_id{0}; const auto cb_id_input = cb_id++; …` (magic CB indices via a runtime counter) | `DFBBinding`s → `DataflowBuffer dfb_input(dfb::input);` etc. Counter and derived ids deleted. |
| `writer_moreh_norm_{w,h,nc}.cpp:19-21` | `uint32_t cb_id{16}; const auto cb_id_output = cb_id++;` | `DataflowBuffer dfb_output(dfb::output);` |
| `moreh_norm_w_kernel.cpp:14-35` · `moreh_norm_h_kernel.cpp:14-35` | `constexpr std::uint8_t input_id = tt::CB::c_in0; constexpr auto cb_x = input_id + 0; …` (7 magic indices) | seven `DFBBinding`s → `dfb::{x,one,mask_*,y,val,cal,reduce}` |
| `moreh_norm_nc_kernel.cpp:12-29` | mutable `std::uint8_t input_id{tt::CB::c_in0}; const auto cb_x = input_id++; …` (5 magic indices) | five `DFBBinding`s → `dfb::{x,one,y,val,cal}` |
| `reader_moreh_norm_w.cpp:45` · `writer_moreh_norm_w.cpp:30` · `reader_moreh_norm_h.cpp:44` · `writer_moreh_norm_h.cpp:27` · `reader_moreh_norm_nc.cpp:34` · `writer_moreh_norm_nc.cpp:27` | `get_tile_size(cb_id_x)` (free function keyed by CB id) | `dfb_x.get_tile_size()` (member getter — whitelist rule 7) |
| all six dataflow kernels + all three compute kernels: `get_arg_val<uint32_t>(i++)` runs | positional RTAs behind a running `int i{0}` counter | named RTAs — `get_arg(args::<name>)`, one per field |
| `..._w_other.cpp:198` · `..._h_other.cpp:190` · `..._nc_other.cpp:174` | `std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, Default)` (CB-id-indexed vector) | `ComputeGen1Config::unpack_modes` — a `Table<DFBSpecName, UnpackMode>` keyed by DFB name; every legacy `Default` maps to `UnpackMode::UnpackToSrc` |

**Semaphore-ID RTAs**: none — the op has no semaphores.
**Page-size third-argument CTAs/RTAs**: none — all six `TensorAccessor` constructions are the 2-arg
form, so nothing to drop.
**Case 2 (raw pointer) bindings**: none — both bindings in all three factories are Case 1, so no
`get_bank_base_address` bridge is used anywhere and the compute-kernel Case-2 block does not apply.

**Deliberately *not* dropped:** the six `*_is_dram` RTAs. They are dead in the kernels (read into a
`const bool` and never used — the `TensorAccessor` already knows the buffer type) but removing them is
a functional change the ops team owns, not port work. Each is preserved as a **named** RTA, still
computed and passed by the host, still read into the same `const bool` local.

---

## Applied Patterns

- [**Self-loop DFB binding**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — the genuine accumulator case, ×8: `val` / `cal` / `reduce` on W and H compute, `val` / `cal` on
  NC compute. Each is bound PRODUCER **and** CONSUMER on the same `KernelSpec` under a **single shared
  accessor name**, so the kernel keeps one `DataflowBuffer` object driving both directions. Re-derived
  from the kernel-touch census, not transcribed: on any node the compute kernel is the *only* toucher
  of each of these (host allocates them; no reader or writer references them), and each does real
  FIFO work in both directions (e.g. `cal` — `reserve_back` at `moreh_norm_w_kernel.cpp:98, 114`,
  `push_back` at `:109, 136`, `wait_front` at `:113, 135`, `pop_front` at `:135`, plus the `reduce<>`
  input at `:140`). One toucher → self-loop; the census agrees with the brief on all eight.
- [**Multi-variant factories**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
  — *shape only, no branching needed.* `moreh_norm` reaches the same end state as the pattern by a
  simpler route: the variant selection already happened one level up, in
  `select_program_factory` (`device/moreh_norm_device_operation.cpp:43-54`), which picks one of three
  *separate* factory structs. So each `create_program_artifacts` builds exactly one `ProgramSpec` with
  no `switch` on a variant attribute. Noted here because the op reads like the pattern's W/H/HW case
  and a porter may expect the branch.
- [**Pass DFB handles directly to LLKs and kernel-lib helpers**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — at every compute call site: `copy_tile(dfb::x, 0, dst0)`, `add_tiles(dfb::val, dfb::cal, 0, 0, dst0)`,
  `binary_op_init_common(dfb::x, dfb::x, dfb::y)`, and in **template-argument** position
  `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::cal, dfb::one, dfb::reduce>(…)`. The
  `constexpr operator uint32_t()` on `DFBAccessor`
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:55`) makes both positions work with no `.id`
  extraction and no temporary wrapper.
- [**Unity-build hygiene for anonymous-namespace symbols**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  — avoided rather than resolved: the three factory `.cpp`s share one unity-build TU, so all typed
  spec-name constants are declared as **function-local** `const`s inside each
  `create_program_artifacts` instead of at anonymous-namespace scope. No shared header, no
  per-factory name prefixes, no collision surface.

**Patterns deliberately *not* applied**, with why:

- **Conditional / optional DFB bindings** — *not applicable.* The mask CB (`c_2`) looks like a
  candidate but both touchers sit behind plain **runtime** `if (do_mask_*)` guards
  (`reader_moreh_norm_w.cpp:36-39`, `moreh_norm_w_kernel.cpp:49-51, 165-167`), never `#ifdef`. A
  runtime `if` still *compiles* both branches, so the binding is unconditional in every
  instantiation and no `#ifdef` gate, no `defines` flag, and no conditional `dfb_bindings` list is
  needed. The `#ifdef`s that *do* exist (`IS_ZERO` / `MINUS_INF`) gate only arithmetic, never a DFB
  reference.
- **Two-toucher DFB → assign 1P+1C** — *not applicable.* The compute pair covers disjoint node sets,
  so no node ever has two touchers of one role. See *Preserved Multiplicity*.
- **Aliased DFBs** / **Same-FIFO aliasing** — *not applicable.* Every legacy `CBDescriptor` has a
  single-element `format_descriptors`, and no kernel or factory aliases one CB index onto another.
- **Sync-free / single-ended CB → self-loop** — *not applicable as such.* All eight self-loops here
  are the genuine accumulator case (real FIFO traffic in both directions), not address-source or
  single-ended CBs.
- **Porting a shared kernel** — *not applicable.* No shared kernel; see *Shared kernels* above.
- **Removing pybound legacy factory entry points** — *not applicable.* No pybind exposure of
  `create_descriptor`.

### Hardware configuration

*Values carried over, not names. Diffed legacy → ported field by field.*

**Data movement.** Legacy uses bare `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` in all
three factories — i.e. the defaulted reader triple (`RISCV_1`, `NOC_0`, `DM_DEDICATED_NOC`) and the
defaulted writer triple (`RISCV_0`, `NOC_1`, `DM_DEDICATED_NOC`). Both resolve to a *default*, so both
take the arch-agnostic TTNN helper:
`create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)` from
`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`. No custom triple
anywhere, so no raw `DataMovementGen1Config`, no `noc_mode` pairing concern, and no legacy
same-NOC misconfiguration to trip the validator.

**Compute — Style A** (the op resolves a TTNN `ComputeKernelConfig` via
`get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config)`). Translate with
`to_compute_hardware_config(arch, operation_attributes.compute_kernel_config)`
(`ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp:76`), which reproduces the
four knobs the legacy `ComputeConfigDescriptor` set:

| legacy `ComputeConfigDescriptor` field | value | Metal 2.0 `ComputeGen1Config` field | transform |
|---|---|---|---|
| `math_fidelity` | from `get_compute_kernel_config_args` | `fpu_math_fidelity` | 1:1 |
| `math_approx_mode` (bool) | same | `sfpu_precision_mode` (`Precision`) | `true` → `Approximate`, `false` → `Precise` |
| `fp32_dest_acc_en` | same | `enable_32_bit_dest` | 1:1 |
| `dst_full_sync_en` | same | `double_buffer_dest` | **inverted** |
| `unpack_to_dest_mode` | `NUM_CIRCULAR_BUFFERS` × `Default` | `unpack_modes` | see below |
| *(not set — `bfp8_pack_precise`)* | default `false` | `bfp_pack_precision_mode` | left default `Precision::Approximate` — the defaults coincide, so nothing to do |
| *(destructured but unused — `packer_l1_acc`)* | — | — | not a Metal-2.0 field; the helper deliberately does not translate it, matching legacy (which never set it) |

**`unpack_modes` — the one field the helper cannot reach, and the one that needs care.** Legacy set an
explicit `UnpackToDestMode::Default` for **every** CB index. `Default` maps to
`UnpackMode::UnpackToSrc`, and `BuildUnpackToDestModeVector`
(`tt_metal/impl/metal2_host_api/program_spec.cpp:2673-2693`) lowers `UnpackToSrc` back to
`UnpackToDestMode::Default` over a `max_cbs`-long all-`Default` vector — so legacy behavior is
reproduced byte-for-byte whether an entry is present or omitted.

The port sets an explicit `UnpackToSrc` entry for **every DFB the compute kernel consumes**, rather
than omitting them. Two reasons, and the choice is deliberate:

1. It is the faithful mirror of legacy, which set `Default` explicitly for all CBs.
2. It satisfies the validator's *required-entry* rule unconditionally. That rule fires when a compute
   kernel consumes a **Float32** DFB with `enable_32_bit_dest = true`
   (`program_spec.cpp:1044-1073`) — reachable here in two independent ways:
   `intermed_data_format` **is** `Float32` exactly when `fp32_dest_acc_en` is set (so `val` / `cal` /
   `reduce` need entries whenever the flag is on), and `cb_data_format` is `Float32` when the input
   dtype is float32 (so `input` / `one` / `mask_*` need them too). Enumerating consumed DFBs covers
   both without a dtype-dependent conditional, and `UnpackToSrc` is always accepted
   (`program_spec.cpp:999-1001`).

Entries are added for **consumed** DFBs only. The `output` DFB is bound PRODUCER-only on compute, and
an entry naming a DFB is legal only if the kernel binds it — producer-only entries would be inert but
pointless, and the required-entry rule does not reach them. The self-looped DFBs *are* consumed (they
carry both roles), so they get entries. Per factory:

- W / H compute: `input`, `one`, `mask_w`/`mask_h`, `val`, `cal`, `reduce` → 6 entries, all `UnpackToSrc`.
- NC compute: `input`, `one`, `val`, `cal` → 4 entries, all `UnpackToSrc`.

**Gen2 is out of scope.** Only the Gen1 alternative is built; both TTNN helpers select the generation
internally, and no `if (arch == QUASAR)` branch is added by this port.

---

## Deferred / Flagged

- **Endpoint census re-derivation: no disagreement with the brief.** All 19 CB→DFB dispositions across
  the three factories were re-derived from the kernel-touch census rather than transcribed, and every
  one matches: 11 plain 1:1 (W `c_0`/`c_1`/`c_2`/`c_16`, H the same four, NC `c_0`/`c_1`/`c_16`) and 8
  self-loops. No dead CB, no multi-binding flag, no 1P+1C reassignment.
- **`TensorParameter` relaxation: none.** The op has no custom `compute_program_hash`, so none can be
  active. Grepped the nine live kernels for `ArgConfig::Runtime` (the pre-migration check in
  `migration_guide.md` → *TensorParameter*): **zero hits**, so no `dynamic_tensor_shape` /
  `match_padded_shape_only` opt-in is required or justified. Strict matching kept everywhere. The
  invoker's readiness-sheet rows (resolving audit Q1) confirm `Is safe to port? == yes` for all three
  factories and name no relaxation.
- **Audit Q1 resolved.** The readiness sheet was unfetchable during the audit; the invoker supplied the
  three rows directly — `ProgramFactoryHOther`, `ProgramFactoryNCOther`, `ProgramFactoryWOther`, all
  `descriptor` concept, all `Is safe to port? = yes`, all `Is able to port? = yes`. Three rows, one per
  factory, matching the code. Nothing in the sheet contradicts the audit's code-derived findings, so
  the port proceeds on all three factories.
- **No new structural findings.** Planning turned up nothing the audit missed: no descriptor type
  outside the audit's Appendix A scope, no construct that resists a binding-token replacement, no
  feature gate that should have fired. Nothing here forces a stop.
- **One planning-time observation worth a report entry, not a code change.** The two compute
  `KernelSpec`s per factory are *byte-identical apart from their `unique_id`* — same source, same
  (empty) CTAs, same defines, same hw_config, same RTA schema. They exist only so that each can sit in
  a different `WorkUnitSpec`. That is the correct shape today (placement is derived from work-unit
  membership, and one spec cannot be in two work units with *different* co-kernels). It is noted for
  downstream as an ergonomics observation, not acted on.
- **`num_units` / RTA values that are node-invariant.** `Wt`, `Ht`, `origin_w`, `origin_h`,
  `num_reduced_tiles_along_dim`, `outer_stride`, `num_inner_tiles`, and both `*_is_dram` flags carry
  the **same value on every node** and are therefore really CRTAs. They stay **RTAs** in this port:
  RTA→CRTA changes dispatch semantics, and the recipe routes it to a later cleanup pass. Recorded in
  the port report under *Open items for downstream*.
