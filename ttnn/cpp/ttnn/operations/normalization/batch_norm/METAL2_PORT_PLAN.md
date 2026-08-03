# Port Plan — `normalization/batch_norm`

Port plan for the two device-operations under `ttnn/cpp/ttnn/operations/normalization/batch_norm/`,
ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope: both factories, one PR** (invoker's explicit scoping; see `METAL2_PORT_BRIEF.md`).
Ported in the brief's recommended order:

| Order | Unit | Device-operation / factory |
|---|---|---|
| 1 | Unit 1 | `RunningStatistics::RunningStatisticsProgramFactory` |
| 2 | Unit 2 | `BatchNormOperation::BatchNormFactory` |

The two device-ops share no factories and no kernels, so neither constrains the other.

---

# Legacy Inventory

## Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** for both — `create_descriptor` returning
  `tt::tt_metal::ProgramDescriptor`
  (`device/running_statistics_device_operation.hpp:36`, `device/batch_norm_device_operation.hpp:39`).
- Variants: **single** each — `program_factory_t` is a one-alternative `std::variant`
  (`running_statistics_device_operation.hpp:42`, `batch_norm_device_operation.hpp:45`).
  No `select_program_factory` in either device-op (correct: the framework returns the sole alternative).
- Custom `compute_program_hash`: **none** in either device-op — already the default reflection-based
  hash. Nothing to delete.
  - `BatchNormOperation::operation_attributes_t::to_hash()`
    (declared `batch_norm_device_operation.hpp:22`, defined `batch_norm_device_operation.cpp:121`) is
    the **ttsl attribute-hash protocol**, not a `compute_program_hash` override. It customizes how
    `operation_attributes_t` hashes *within* the default reflection hash; tensor args (hence
    `TensorSpec`) are still folded in. The sanctioned custom-hash deletion does not apply.
    **Left untouched** — invoker confirmed the recipe default.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

---

## Variant: Unit 1 — `RunningStatisticsProgramFactory`

Source: `device/running_statistics_program_factory.cpp` (468 lines).
All three `KernelDescriptor`s are placed on `all_device_cores`
(`CoreRangeSet(CoreRange({0,0},{grid.x-1, grid.y-1}))`, `:185`).

Config predicates computed at `:150-179`:

| Predicate | Definition |
|---|---|
| `running_mean_has_value` / `running_var_has_value` | `tensor_args.running_{mean,var}.has_value()` |
| `any_float32` | any of the 5 involved dtypes is `Float32` |
| `interm_data_format` | `any_float32 ? Float32 : a_data_format` |
| `running_stat_data_format` | `running_mean ? d : (running_var ? e : Float16_b)` |
| `stat_format_needs_typecast` | `interm == Float32 && running_stat_data_format != Float32` |
| `needs_mean_typecast` | `running_mean_has_value && stat_format_needs_typecast` |
| `needs_var_typecast` | `running_var_has_value && stat_format_needs_typecast` |
| `writer_updated_m_cb` | `needs_mean_typecast ? c_12 : c_7` (`:283-297`) |
| `writer_updated_v_cb` | `needs_var_typecast ? c_13 : c_8` (`:284-310`) |

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_running_statistics.cpp` | `all_device_cores` | `[0]` `batch_mean_cb`=c_0 · `[1]` `momentum_cb`=c_5 · `[2]` `one_cb`=c_6 · `[3..]` `TensorAccessorArgs(batch_mean)` · `[+1]` `any_float32` (`:346-352`) | none | 11 per core, node-first (`:85-97`): `packed_scalar_momentum`, `batch_mean` **`Buffer*`**, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `n_stride`, `c_stride`, `cN`, `cC`, `cHt`, `cWt` | none | none | absent → **O2** (DM default) | `ReaderConfigDescriptor{}` (`:379`) |
| writer | `device/kernels/dataflow/writer_running_statistics.cpp` | `all_device_cores` | `[0]` `running_mean_has_value` · `[1]` `running_var_has_value` · `[2]` `batch_var_cb`=c_1 · `[3]` `output_cb`=c_2 · `[4]` `old_running_mean_cb`=c_3 · `[5]` `old_running_var_cb`=c_4 · `[6]` `writer_updated_m_cb` · `[7]` `writer_updated_v_cb` · `[8..]` `TensorAccessorArgs` ×4 (batch_var, output, running_mean\|**nullptr**, running_var\|**nullptr**) · `[+1]` `old_stat_is_fp32` (`:354-370`) | none | 13 per core (`:107-121`): `batch_var` **`Buffer*`**, `running_mean_arg` (**`Buffer*`** or literal `0u`), `running_var_arg` (ditto), `output` **`Buffer*`**, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `n_stride`, `c_stride`, `cN`, `cC`, `cHt`, `cWt` | none | none | absent → **O2** | `WriterConfigDescriptor{}` (`:388`) |
| compute | **runtime-selected**, `fmt::format` at `:438-440`: `device/kernels/compute/running_statistics_{sfpu_kernel\|kernel}.cpp`, selector `(fp32_dest_acc_en \|\| any_float32)` | `all_device_cores` | 19 (`:416-435`): `[0]` `running_mean_has_value` · `[1]` `running_var_has_value` · `[2]` c_0 · `[3]` c_1 · `[4]` c_2 · `[5]` c_3 · `[6]` c_4 · `[7]` c_7 · `[8]` c_8 · `[9]` c_5 · `[10]` c_6 · `[11]` c_9 · `[12]` c_10 · `[13]` c_11 · `[14]` `writer_updated_m_cb` · `[15]` `writer_updated_v_cb` · `[16]` `stat_format_needs_typecast` · `[17]` `DataFormat::Float32` · `[18]` `tc_out_fmt` | none | 3 per core (`:126-127`): `num_tiles_per_core`, `freq`, `counter` | none | none | absent → **O3** (`ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:444-450`) |

Compute-config provenance: **Style A** — `get_compute_kernel_config_args(device->arch(),
operation_attributes.compute_kernel_config)` at `:391-392`, over a config resolved by
`batch_norm::utils::resolve_compute_kernel_config`. `bfp8_pack_precise` left at its default.

Idle-core zero-fill (`:72-78`): cores in neither work group get `KernelDescriptor::CoreRuntimeArgs`
of all-zeros sized `num_reader_args = 11`, `num_writer_args = 13`, `num_kernel_args = 3` (`:60-62`).

### CBs

All 12–14 CBs are single-`CBFormatDescriptor` plain `CBDescriptor`s on `all_device_cores`, with
`total_size = page_size * 2` (`num_tiles_per_cb = b_num_tiles_per_cb = 2`). No `tile` field is ever
set → `tile_format_metadata` is `nullopt` throughout. No `address_offset`, no
`global_circular_buffer`, no `set_globally_allocated_address`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | role |
|---|---|---|---|---|---|---|
| c_0 | `a_tile*2` | all | `a_data_format` | `a_tile` | — | batch_mean |
| c_1 | `b_tile*2` | all | `b_data_format` | `b_tile` | — | batch_var |
| c_2 | `c_tile*2` | all | `c_data_format` | `c_tile` | — | out0 (output) |
| c_3 | `d_tile*2` | all | `d_data_format` | `d_tile` | — | old_running_mean |
| c_4 | `e_tile*2` | all | `e_data_format` | `e_tile` | — | old_running_var |
| c_5 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | momentum |
| c_6 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | one |
| c_7 | `(needs_mean_typecast?interm:d)_tile*2` | all | `needs_mean_typecast?interm:d` | ditto | — | updated_m (FP32 staging when typecast) |
| c_8 | `(needs_var_typecast?interm:e)_tile*2` | all | `needs_var_typecast?interm:e` | ditto | — | updated_v |
| c_9 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | tmp1 |
| c_10 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | tmp2 |
| c_11 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | tmp3 |
| c_12 | `d_tile*2` | all | `d_data_format` | `d_tile` | — | writer-facing updated_m — **only when `needs_mean_typecast`** |
| c_13 | `e_tile*2` | all | `e_data_format` | `e_tile` | — | writer-facing updated_v — **only when `needs_var_typecast`** |

### Semaphores

none — `grep -n '[Ss]emaphore'` over the whole op directory is empty.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `running_statistics_program_factory.cpp:351` | `tensor_args.batch_mean` | reader slot 1 (`Buffer*`) |
| `running_statistics_program_factory.cpp:364` | `tensor_args.batch_var` | writer slot 0 (`Buffer*`) |
| `running_statistics_program_factory.cpp:365` | `tensor_return_value` (output) | writer slot 3 (`Buffer*`) |
| `running_statistics_program_factory.cpp:366` | `tensor_args.running_mean` (or `nullptr`) | writer slot 1 (`Buffer*` or `0u`) |
| `running_statistics_program_factory.cpp:368` | `tensor_args.running_var` (or `nullptr`) | writer slot 2 (`Buffer*` or `0u`) |

Kernel-side: `reader_running_statistics.cpp:29`, `writer_running_statistics.cpp:36-39`.
All 5 are **Case 1** (address feeds a `TensorAccessor`, all access via the accessor). No
`get_bank_base_address` bridge needed. All sites are already **2-arg** — no page-size third argument.

### Work split

- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, /*row_major=*/true)`
  (`:50-57`), inside `populate_runtime_arguments`.
- `num_cores` / `all_cores` are **discarded** (`_unused_*`); only `core_group_1`, `core_group_2` and
  the two per-group counts are used, and only to pick each core's `num_tiles_per_core` **runtime**
  value. Every kernel is placed on `all_device_cores` regardless.
- `cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major)` — the full grid.

### Shared kernels

**none.** All four kernel sources live in this op's own `device/kernels/` tree; `grep -rl` per kernel
filename across `ttnn/cpp/ttnn/operations/` returns only this op's own factory (the two compute paths
return no hit at all because the path is assembled with `fmt::format` — confirmed by reading the
format string at `:438-440`). The two device-ops share no kernel with each other, so the single-PR
scoping introduces no intra-op sharing either. No `_metal2` fork exists or is needed.

---

## Variant: Unit 2 — `BatchNormFactory`

Source: `device/batch_norm_program_factory.cpp` (418 lines). Same placement shape:
all three kernels on `all_device_cores` (`:188`).

Config predicates (`:153-182`): `weight_has_value`, `bias_has_value`, `any_float32` (over 6 dtypes),
`interm_data_format = any_float32 ? Float32 : a`, `needs_output_typecast = (interm == Float32 &&
c_data_format != Float32)`, `writer_output_cb = needs_output_typecast ? c_9 : c_2` (`:226-239`),
`param_data_format = weight ? e : (bias ? f : Float16_b)` (`:326-327`).

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_batch_norm.cpp` | `all_device_cores` | `[0]` `input_cb`=c_0 · `[1]` `eps_cb`=c_4 · `[2..]` `TensorAccessorArgs(input)` · `[+1]` `any_float32` (`:303-308`) | none | 11 per core (`:87-99`): `packed_scalar_eps`, `input` **`Buffer*`**, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `n_stride`, `c_stride`, `cN`, `cC`, `cHt`, `cWt` | none | none | absent → **O2** | `ReaderConfigDescriptor{}` (`:337`) |
| writer | `device/kernels/dataflow/writer_batch_norm.cpp` | `all_device_cores` | `[0]` `weight_has_value` · `[1]` `bias_has_value` · `[2]` `batch_mean_cb`=c_1 · `[3]` `writer_output_cb` · `[4]` `batch_var_cb`=c_3 · `[5]` `weight_cb`=c_5 · `[6]` `bias_cb`=c_6 · `[7..]` `TensorAccessorArgs` ×5 (batch_mean, output, batch_var, weight\|**nullptr**, bias\|**nullptr**) · `[+1]` `batch_stat_is_fp32` · `[+2]` `param_is_fp32` (`:310-328`) | none | 14 per core (`:109-124`): `batch_mean` **`Buffer*`**, `batch_var` **`Buffer*`**, `weight_arg`, `bias_arg`, `output` **`Buffer*`**, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `n_stride`, `c_stride`, `cN`, `cC`, `cHt`, `cWt` | none | none | absent → **O2** | `WriterConfigDescriptor{}` (`:346`) |
| compute | **runtime-selected**, `fmt::format` at `:388-390`: `device/kernels/compute/batch_norm_{sfpu_kernel\|kernel}.cpp`, selector `(fp32_dest_acc_en \|\| any_float32)` | `all_device_cores` | 15 (`:370-385`): `[0]` `weight_has_value` · `[1]` `bias_has_value` · `[2]` c_0 · `[3]` c_1 · `[4]` c_2 · `[5]` c_3 · `[6]` c_4 · `[7]` c_7 · `[8]` c_5 · `[9]` c_8 · `[10]` c_6 · `[11]` `writer_output_cb` · `[12]` `needs_output_typecast` · `[13]` `DataFormat::Float32` · `[14]` `tc_out_fmt` | none | 3 per core (`:129-130`): `num_tiles_per_core`, `freq`, `counter` | none | none | absent → **O3** | `ComputeConfigDescriptor{…}` (`:394-400`) |

Compute-config provenance: **Style A** at `:349-350`. `bfp8_pack_precise` left at its default.

Idle-core zero-fill (`:73-79`): `num_reader_args = 11`, `num_writer_args = 14`, `num_kernel_args = 3`
(`:61-63`).

### CBs

Same shape as Unit 1: plain single-format `CBDescriptor`s on `all_device_cores`, `total_size =
page_size * 2`, no `tile`, no borrowed memory, no GlobalCB.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | role |
|---|---|---|---|---|---|---|
| c_0 | `a_tile*2` | all | `a_data_format` | `a_tile` | — | input |
| c_1 | `b_tile*2` | all | `b_data_format` | `b_tile` | — | batch_mean |
| c_2 | `(needs_output_typecast?interm:c)_tile*2` | all | `needs_output_typecast?interm:c` | ditto | — | output_0 (FP32 staging when typecast) |
| c_3 | `d_tile*2` | all | `d_data_format` | `d_tile` | — | batch_var |
| c_4 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | eps |
| c_5 | `e_tile*2` | all | `e_data_format` | `e_tile` | — | weight |
| c_6 | `f_tile*2` | all | `f_data_format` | `f_tile` | — | bias |
| c_7 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | den = `1/sqrt(var+eps)` |
| c_8 | `interm_tile*2` | all | `interm_data_format` | `interm_tile` | — | temp_1 |
| c_9 | `c_tile*2` | all | `c_data_format` | `c_tile` | — | writer-facing output — **only when `needs_output_typecast`** |

### Semaphores

none.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `batch_norm_program_factory.cpp:307` | `tensor_args.input` | reader slot 1 (`Buffer*`) |
| `batch_norm_program_factory.cpp:319` | `tensor_args.batch_mean` | writer slot 0 (`Buffer*`) |
| `batch_norm_program_factory.cpp:320` | `tensor_return_value` (output) | writer slot 4 (`Buffer*`) |
| `batch_norm_program_factory.cpp:321` | `tensor_args.batch_var` | writer slot 1 (`Buffer*`) |
| `batch_norm_program_factory.cpp:322` | `tensor_args.weight` (or `nullptr`) | writer slot 2 (`Buffer*` or `0u`) |
| `batch_norm_program_factory.cpp:324` | `tensor_args.bias` (or `nullptr`) | writer slot 3 (`Buffer*` or `0u`) |

Kernel-side: `reader_batch_norm.cpp:27`, `writer_batch_norm.cpp:37-41`. All 6 **Case 1**, all 2-arg.

### Work split

Identical shape to Unit 1: `split_work_to_cores(compute_with_storage_grid_size, num_output_tiles,
true)` at `:50-57`, `num_cores`/`all_cores` discarded, per-group counts delivered purely as RTAs,
all kernels on `all_device_cores`.

### Shared kernels

**none** — same census result as Unit 1.

---

## Flags

Observations the inventory noticed but did not classify. None is port work; each is carried forward
unchanged and re-stated in `METAL2_PORT_REPORT.md`.

- **Unreferenced kernel files: none.** All eight kernels are instantiated by their device-op's factory.
- **Dead RTA slots** (audit anomaly A1). Both factories emit more RTAs than their kernels read:
  reader emits 11, kernel reads 0–8 (`cHt`, `cWt` dead); RunningStatistics writer emits 13, reads
  0–10; BatchNorm writer emits 14, reads 0–11; RunningStatistics compute emits 3, **both** compute
  sources read only slot 0 (`freq`, `counter` dead — the BatchNorm compute kernels read all three).
  A named schema makes the surplus explicit, so the dead names are **not declared** post-port.
- **Dead CTA slots in the non-SFPU compute sources** (A2). The host CTA list is built for the SFPU
  superset: `running_statistics_kernel.cpp` reads CTAs 0–13 of 19; `batch_norm_kernel.cpp` reads
  0–10 of 15. **Not pruned** — the sibling SFPU source, selected from the same `KernelSpec`, reads them.
- **`push_back` with no matching `reserve_back` on RunningStatistics `c_2`** (A3):
  `running_statistics_kernel.cpp:57-59` packs into `dfb_out0` and `push_back`s without ever
  reserving; the SFPU sibling *does* reserve (`running_statistics_sfpu_kernel.cpp:95`). A legacy
  FIFO-protocol asymmetry, **carried forward unchanged** (the recipe forbids adding a
  `reserve`/`pop` to "balance" a FIFO).
- **Duplicated `extract_shape_dims` / `populate_runtime_arguments`** (A4) between the two factories,
  both inside `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`. Both copies land in one diff
  under the single-PR scoping. **Not de-duplicated.**
- **`TensorAccessorArgs(nullptr)` for absent optionals** (A5) —
  `running_statistics_program_factory.cpp:366,368`, `batch_norm_program_factory.cpp:322,324`, with
  the kernels constructing a `TensorAccessor` over address `0` unconditionally. The port **removes
  this idiom entirely** (conditional `TensorBinding` + `#ifdef`); that removal is port work, not a
  bundled fix.
- **`RunningStatistics` mutates its inputs in place** (A6) — `writer_running_statistics.cpp:103-110,
  132-139` writes updated statistics back into the `running_mean` / `running_var` **input** tensors,
  while the declared output receives a duplicate of one stat. Legal; **ported as-is**. One
  `TensorParameter` with one `TensorBinding` serves both directions.
- **Kernel code tracks DFB ids in runtime `uint32_t` variables.** `batch_norm_kernel.cpp:31-32` and
  `batch_norm_sfpu_kernel.cpp:42-43` pick `dfb_affine_or_out` / `dfb_scaled_output` with a **runtime**
  ternary and construct `DataflowBuffer` from the result; the SFPU sources thread a mutable
  `last_srca_dfb` id through returns and parameters; `batchnorm_bcast_tiles(...)` takes all thirteen
  ids as plain `uint32_t` parameters. All of this keeps working (`DFBAccessor::operator uint32_t()`
  is `constexpr`; `DataflowBuffer` has a `uint16_t` constructor) and is **left alone** — rewriting a
  helper signature or the `last_srca_dfb` threading would be kernel-logic surgery.
- **`ttnn/operations/cb_utils.hpp`** is included by both factories but its only symbol (`create_cb`)
  is never used; the include is load-bearing today only because it transitively pulls
  `host_api.hpp` → `bfloat16.hpp` for `pack_two_bfloat16_into_uint32`. The port drops the legacy-CB
  include and adds `<tt-metalium/bfloat16.hpp>` explicitly.

---

# TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — `create_program_artifacts`
  returning `ttnn::device_operation::ProgramArtifacts`, for **both** factories.
  `op_owned_tensors` left defaulted (neither factory allocates device tensors beyond declared io).
- **Custom `compute_program_hash`**: **none** — nothing to delete in either device-op.
  `BatchNormOperation::…::to_hash()` is a different mechanism and stays (see Legacy Inventory).
- **Device-op-class edits forced — exactly one per header:**
  - `running_statistics_device_operation.hpp`: `-#include <tt-metalium/program_descriptors.hpp>` →
    `+#include "ttnn/metal_v2_artifacts.hpp"`; `create_descriptor` → `create_program_artifacts`
    returning `ttnn::device_operation::ProgramArtifacts`.
  - `batch_norm_device_operation.hpp`: the same two edits.
  - **No pybind removal** — `batch_norm_nanobind.cpp:77` binds only the user-facing `ttnn::batch_norm`;
    no factory entry point is exposed.
- **Implementation notes**: both factories keep their existing `namespace { namespace
  CMAKE_UNIQUE_NAMESPACE { … } }` private helpers. Because both `.cpp` files land in the same
  unity-build translation unit, **every** `DFBSpecName` / `KernelSpecName` / `TensorParamName`
  constant is declared **function-local inside `create_program_artifacts`** — no
  anonymous-namespace constants, no shared header.

---

# Planned Spec Shape

## Unit 1 — `RunningStatisticsProgramFactory`

- **KernelSpecs (3)** — one per legacy `KernelDescriptor`:
  `READER` (`reader`), `WRITER` (`writer`), `COMPUTE` (`compute`). The compute spec's `.source` is
  the same runtime-selected `fmt::format` expression as legacy; **both** selectable sources bind
  this one spec.
- **DataflowBufferSpecs (12, +1 per typecast flag → up to 14)** — one per legacy `CBDescriptor`.
  No legacy CB has multi-element `format_descriptors`, so there are **no aliased DFBs**
  (`advanced_options.alias_with` unused). No `borrowed_from`. No
  `allow_instance_multi_binding` anywhere.

  | DFBSpecName | legacy CB | `entry_size` | `num_entries` | `data_format_metadata` | declared |
  |---|---|---|---|---|---|
  | `batch_mean` | c_0 | `a_tile` | 2 | `a` | always |
  | `batch_var` | c_1 | `b_tile` | 2 | `b` | always |
  | `out0` | c_2 | `c_tile` | 2 | `c` | always |
  | `old_running_mean` | c_3 | `d_tile` | 2 | `d` | always |
  | `old_running_var` | c_4 | `e_tile` | 2 | `e` | always |
  | `momentum` | c_5 | `interm_tile` | 2 | `interm` | always |
  | `one` | c_6 | `interm_tile` | 2 | `interm` | always |
  | `updated_m` | c_7 | `needs_mean_typecast?interm:d`-tile | 2 | ditto | always |
  | `updated_v` | c_8 | `needs_var_typecast?interm:e`-tile | 2 | ditto | always |
  | `tmp1` | c_9 | `interm_tile` | 2 | `interm` | always |
  | `tmp2` | c_10 | `interm_tile` | 2 | `interm` | always |
  | `tmp3` | c_11 | `interm_tile` | 2 | `interm` | always |
  | `writer_updated_m` | c_12 | `d_tile` | 2 | `d` | **iff `needs_mean_typecast`** |
  | `writer_updated_v` | c_13 | `e_tile` | 2 | `e` | **iff `needs_var_typecast`** |

  `tile_format_metadata` left `nullopt` everywhere (legacy never set `format_descriptors[i].tile`).

- **SemaphoreSpecs**: none — the op uses no semaphores.
- **TensorParameters (5)** — one per distinct originating tensor:
  `batch_mean`, `batch_var`, `output` (always) and `running_mean`, `running_var`
  (**conditional** — declared only when the optional has a value).
- **WorkUnitSpecs (1)**: `{name = "main", kernels = {READER, WRITER, COMPUTE}, target_nodes =
  all_device_cores}`. All three legacy `KernelDescriptor`s share one `core_ranges`, so one WU
  reproduces placement exactly and satisfies the local-DFB identical-WU-membership invariant.
- **Op-owned tensors**: none.

### DFB endpoint dispositions (re-derived from the kernel-touch census, not transcribed)

Census run per DFB, per node, per config, over distinct kernel instances that FIFO-produce,
FIFO-consume, or raw-touch the buffer. All three kernels sit on the single node set
`all_device_cores`, so every node sees exactly one reader, one writer and one compute instance —
**no dual-instance work-split and no same-source multiplicity anywhere.**

| DFB | reader | writer | compute | disposition |
|---|---|---|---|---|
| `batch_mean` | PRODUCER (`reader_running_statistics.cpp:73-76`) | — | CONSUMER (`sfpu:94,196`) | 1P+1C |
| `batch_var` | — | **PRODUCER** (`writer_running_statistics.cpp:79-82`) | CONSUMER (`sfpu:219,237`) | 1P+1C |
| `out0` | — | CONSUMER (`writer:144-148`) | PRODUCER (`sfpu:95,294`; `plain:57-59`) | 1P+1C |
| `old_running_mean` | — | **PRODUCER** (`writer:86-99`) | CONSUMER (`sfpu:138,156`) | 1P+1C |
| `old_running_var` | — | **PRODUCER** (`writer:115-128`) | CONSUMER (`sfpu:241,258`) | 1P+1C |
| `momentum` | PRODUCER (`reader:59-67`) | — | CONSUMER (`sfpu:86,296`) | 1P+1C |
| `one` | PRODUCER via `fill_cb_with_value` (`reader:56`) | — | CONSUMER (`sfpu:87,297`) | 1P+1C |
| `tmp1` / `tmp2` / `tmp3` | — | — | PRODUCER **and** CONSUMER | **self-loop** ×3 |
| `updated_m` | — | CONSUMER **iff `!needs_mean_typecast`** | PRODUCER; **also** CONSUMER iff `needs_mean_typecast` | **config-flip**: 1P+1C ↔ self-loop |
| `updated_v` | — | CONSUMER iff `!needs_var_typecast` | PRODUCER; also CONSUMER iff `needs_var_typecast` | **config-flip** |
| `writer_updated_m` | — | CONSUMER | PRODUCER | 1P+1C — exists only when `needs_mean_typecast` |
| `writer_updated_v` | — | CONSUMER | PRODUCER | 1P+1C — only when `needs_var_typecast` |

**My census agrees with the brief on all 12–14 DFBs.** Notes:

- The **writer** kernel is a *producer* on `batch_var`, `old_running_mean` and `old_running_var` (it
  reads tensor memory into them) as well as the consumer of `out0` and the updated stats. Bound by
  what the body does, not by the kernel's name.
- `updated_m` / `updated_v` flip because on the typecast path **compute itself** re-reads its own
  staging buffer (`running_statistics_sfpu_kernel.cpp:18-41`, `maybe_typecast_stat`) and the writer
  drains `writer_updated_*` instead. The typecast path is reachable **only** through the SFPU source
  (`stat_format_needs_typecast` requires `interm == Float32` ⇒ `any_float32` ⇒ SFPU selection),
  which the non-SFPU source confirms from the other side by never reading CTAs 14–18.
- `old_running_mean` / `old_running_var` are bound **unconditionally** even when the optional tensor
  is absent: the host allocates the CBs on every config (`:222-241`) and the kernels construct
  `DataflowBuffer` objects for them outside the conditional
  (`writer_running_statistics.cpp:46-47`, `running_statistics_sfpu_kernel.cpp:72-73`). Dropping them
  would shrink the op's L1 footprint relative to legacy — a functional change — and would fail to
  compile. Census is a plain 1P+1C.
- **No DFB is both self-looped and multi-bound**, and `allow_instance_multi_binding` is not set
  anywhere. Every DFB has at most one producing and one consuming kernel instance per node.

### Same-FIFO aliasing: `writer_updated_*` on the typecast-off path

When `needs_mean_typecast` is false, legacy sets `writer_updated_m_cb = updated_m_cb` (`:283`) — one
CB reached through two kernel-side names (`dfb_id_updated_running_mean` on the writer,
`dfb_writer_updated_mean` on compute). This is
[Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md),
**not** `alias_with`: one DFB, one FIFO, two names. Handled two different ways, per kernel:

- **Writer** — resolved entirely **on the host**. The writer's accessor name is the constant
  `new_mean`, and the host binds it to `needs_mean_typecast ? WRITER_UPDATED_M : UPDATED_M`. The
  kernel needs no `#ifdef` at all and keeps one `DataflowBuffer dfb_new_mean(dfb::new_mean)`.
- **Compute** — needs a **kernel-side handle alias**, because compute binds `UPDATED_M` under its own
  accessor name already and the validator rejects a second accessor name for the same DFB on one
  kernel. Gated with the path-dependent variant of the pattern:

  ```cpp
  #ifdef MEAN_NEEDS_TYPECAST
  constexpr auto dfb_writer_updated_mean = dfb::writer_updated_m;
  #else
  constexpr auto dfb_writer_updated_mean = dfb_updated_running_mean;
  #endif
  ```

  The define is emitted from the host on the **compute** kernel only — the one kernel that references
  `dfb::writer_updated_m`.

## Unit 2 — `BatchNormFactory`

- **KernelSpecs (3)**: `READER`, `WRITER`, `COMPUTE` (compute source runtime-selected, both sources
  bind the one spec).
- **DataflowBufferSpecs (9, +1 → 10)**:

  | DFBSpecName | legacy CB | `entry_size` | `num_entries` | `data_format_metadata` | declared |
  |---|---|---|---|---|---|
  | `input` | c_0 | `a_tile` | 2 | `a` | always |
  | `batch_mean` | c_1 | `b_tile` | 2 | `b` | always |
  | `output_0` | c_2 | `needs_output_typecast?interm:c`-tile | 2 | ditto | always |
  | `batch_var` | c_3 | `d_tile` | 2 | `d` | always |
  | `eps` | c_4 | `interm_tile` | 2 | `interm` | always |
  | `weight` | c_5 | `e_tile` | 2 | `e` | always |
  | `bias` | c_6 | `f_tile` | 2 | `f` | always |
  | `den` | c_7 | `interm_tile` | 2 | `interm` | always |
  | `temp_1` | c_8 | `interm_tile` | 2 | `interm` | always |
  | `writer_output` | c_9 | `c_tile` | 2 | `c` | **iff `needs_output_typecast`** |

- **SemaphoreSpecs**: none.
- **TensorParameters (6)**: `input`, `batch_mean`, `batch_var`, `output` (always); `weight`, `bias`
  (**conditional**).
- **WorkUnitSpecs (1)**: `{"main", {READER, WRITER, COMPUTE}, all_device_cores}`.
- **Op-owned tensors**: none.

### DFB endpoint dispositions (re-derived)

| DFB | reader | writer | compute | disposition |
|---|---|---|---|---|
| `input` | PRODUCER (`reader_batch_norm.cpp:66-69`) | — | CONSUMER (`sfpu:89,114`) | 1P+1C |
| `batch_mean` | — | **PRODUCER** (`writer_batch_norm.cpp:85-93`) | CONSUMER (`sfpu:80,187`) | 1P+1C |
| `batch_var` | — | **PRODUCER** (`writer:96-105`) | CONSUMER (`sfpu:58,78`) | 1P+1C |
| `eps` | PRODUCER (`reader:46-54`) | — | CONSUMER (`sfpu:235,274`) | 1P+1C |
| `weight` | — | **PRODUCER** (`writer:107-117`) | CONSUMER (`sfpu:83,190`) | 1P+1C, **bind unconditionally** |
| `bias` | — | **PRODUCER** (`writer:119-129`) | CONSUMER (`sfpu:86,193`) | 1P+1C, **bind unconditionally** |
| `den` | — | — | PRODUCER **and** CONSUMER | **self-loop** |
| `temp_1` | — | — | PRODUCER and CONSUMER (via the runtime aliases) | **self-loop** |
| `output_0` | — | CONSUMER iff `!needs_output_typecast` | PRODUCER; **also** CONSUMER iff `needs_output_typecast` | **config-flip** |
| `writer_output` | — | CONSUMER | PRODUCER | 1P+1C — only when `needs_output_typecast` |

**Agrees with the brief.** Notes:

- `temp_1` is reached only through the **runtime** aliases `dfb_affine_or_out` / `dfb_scaled_output`
  (`batch_norm_kernel.cpp:31-32`, `batch_norm_sfpu_kernel.cpp:42-43`). When neither weight nor bias
  is present those aliases resolve to `output_0` and `temp_1` is untouched at runtime — but
  `dfb_tmp_1_obj` is still constructed unconditionally (`batch_norm_kernel.cpp:40`), so it is
  referenced in compiled code in every config. Bound as a compute self-loop in **every** config.
- `weight` / `bias` bound unconditionally for the same reason as Unit 1's `old_running_*`: the host
  allocates c_5/c_6 on every config (`:260-279`) and the writer constructs their `DataflowBuffer`s
  outside the `if constexpr` (`writer_batch_norm.cpp:49-50`). The `batch_norm` compute kernels
  additionally gate on a plain **runtime** `if` (`:65,93`), so both branches compile.
- `output_0` flips because on the typecast path compute re-reads it to typecast into `writer_output`
  (`batch_norm_sfpu_kernel.cpp:163-185`) and the writer drains `writer_output`.
- No multi-binding flag; no dead CB.

### Same-FIFO aliasing: `writer_output_cb` on the typecast-off path

Identical shape to Unit 1. Writer: host binds accessor `dst` to
`needs_output_typecast ? WRITER_OUTPUT : OUTPUT_0` — no kernel `#ifdef`. Compute: kernel-side handle
alias gated on `NEEDS_OUTPUT_TYPECAST`, aliasing `dfb_output_0` on the off-path.

---

# Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Both factories call `split_work_to_cores` but place
every `KernelDescriptor` on the single `all_device_cores` range and deliver the per-group counts as
**runtime args only**. There is no second `KernelDescriptor` per core group and no per-group CTA, so
there is nothing to preserve — and correspondingly no
[Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
risk: nothing was ever a per-group CTA.

Idle cores (in neither work group) keep their legacy **zero-filled** RTAs. Metal 2.0 requires every
named RTA to be set on every node the kernel runs on, and since the kernels run on the whole grid the
zero-fill is exactly what satisfies that requirement. The existing per-core loop is kept as-is and
bridged with `AddRuntimeArgsForNode`; the loop is **not** inverted to name-first.

---

# Dropped Plumbing

## Unit 1 — `RunningStatisticsProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `running_statistics_program_factory.cpp:351` | `TensorAccessorArgs(batch_mean.buffer()).append_to(reader_cta)` | `TensorParameter{batch_mean}` + reader `TensorBinding{accessor "src"}` |
| `:364` | `TensorAccessorArgs(batch_var.buffer()).append_to(writer_cta)` | `TensorParameter{batch_var}` + writer `TensorBinding{"src"}` |
| `:365` | `TensorAccessorArgs(output.buffer())…` | `TensorParameter{output}` + writer `TensorBinding{"dst"}` |
| `:366` | `TensorAccessorArgs(running_mean ? …->buffer() : nullptr)…` | **conditional** `TensorParameter{running_mean}` + writer `TensorBinding{"old_running_mean"}`, gated by `RUNNING_MEAN_HAS_VALUE` define |
| `:368` | `TensorAccessorArgs(running_var ? …->buffer() : nullptr)…` | **conditional** `TensorParameter{running_var}` + writer `TensorBinding{"old_running_var"}`, gated by `RUNNING_VAR_HAS_VALUE` |
| reader RTA slot 1 (`:88`) | `batch_mean_tensor.buffer()` (`Buffer*`) | `TensorBinding` (address auto-injected) |
| writer RTA slots 0,1,2,3 (`:109-112`) | `batch_var.buffer()`, `running_mean_arg`, `running_var_arg`, `c.buffer()` | four `TensorBinding`s |
| reader CTA 0,1,2 (`:347-349`) | `batch_mean_cb`, `momentum_cb`, `one_cb` magic indices | `DFBBinding`s → `dfb::src`, `dfb::momentum`, `dfb::one` |
| writer CTA 2,3,4,5,6,7 (`:357-362`) | `batch_var_cb`, `output_cb`, `old_running_mean_cb`, `old_running_var_cb`, `writer_updated_m_cb`, `writer_updated_v_cb` | `DFBBinding`s → `dfb::src`, `dfb::dst`, `dfb::old_running_mean`, `dfb::old_running_var`, `dfb::new_mean`, `dfb::new_var` |
| compute CTA 2–15 (`:419-432`) | 14 magic CB indices | `DFBBinding`s (12 unconditional + `writer_updated_*` when typecast, the latter reached through the `#ifdef`-gated handle alias) |
| writer CTA 0,1 (`:355-356`) | `running_mean_has_value`, `running_var_has_value` — read **only** by `if constexpr` at `writer_running_statistics.cpp:84,113` | promoted to `compiler_options.defines` `RUNNING_MEAN_HAS_VALUE` / `RUNNING_VAR_HAS_VALUE` (the gate must move to the preprocessor because the guarded block now names a conditionally-bound `tensor::`) |
| reader RTA slots 9,10; writer slots 11,12; compute slots 1,2 | dead RTAs the kernels never read (anomaly A1) | **not declared** in the named schema |
| reader CTA (last), writer CTA (last), compute CTA 0,1,16,17,18 | positional scalars | **named** CTAs — see below |

Named CTAs (non-CB scalars only):

- reader: `fill_momentum_fp32`
- writer: `old_stat_is_fp32`
- compute: `old_running_mean_has_value`, `old_running_var_has_value`, `stat_needs_typecast`,
  `tc_in_fmt`, `tc_out_fmt` — the compute sources read these as CTAs (the `has_value` pair also feeds
  `needs_{mean,var}_typecast`), so they stay named args there. The **superset** the SFPU source reads
  is declared; the non-SFPU source ignores the five it does not read.

Named RTAs:

- reader (8): `momentum`, `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`
- writer (7): `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`
- compute (1): `num_tiles`

No CRTAs, and **no varargs** — every RTA in all four kernels is read once at a source-literal index
as a distinct field.

## Unit 2 — `BatchNormFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `batch_norm_program_factory.cpp:307` | `TensorAccessorArgs(input.buffer())…` | `TensorParameter{input}` + reader `TensorBinding{"src"}` |
| `:319` | `TensorAccessorArgs(batch_mean.buffer())…` | `TensorParameter{batch_mean}` + writer `TensorBinding{"src"}` |
| `:320` | `TensorAccessorArgs(output.buffer())…` | `TensorParameter{output}` + writer `TensorBinding{"dst"}` |
| `:321` | `TensorAccessorArgs(batch_var.buffer())…` | `TensorParameter{batch_var}` + writer `TensorBinding{"batch_var"}` |
| `:322` | `TensorAccessorArgs(weight ? …->buffer() : nullptr)…` | **conditional** `TensorParameter{weight}` + writer `TensorBinding{"weight"}`, gated by `WEIGHT_HAS_VALUE` |
| `:324` | `TensorAccessorArgs(bias ? …->buffer() : nullptr)…` | **conditional** `TensorParameter{bias}` + writer `TensorBinding{"bias"}`, gated by `BIAS_HAS_VALUE` |
| reader RTA slot 1 (`:90`) | `input_tensor.buffer()` | `TensorBinding` |
| writer RTA slots 0–4 (`:111-115`) | `batch_mean.buffer()`, `batch_var.buffer()`, `weight_arg`, `bias_arg`, `c.buffer()` | five `TensorBinding`s |
| reader CTA 0,1 (`:304-305`) | `input_cb`, `eps_cb` | `dfb::src`, `dfb::eps` |
| writer CTA 2,3,4,5,6 (`:313-317`) | `batch_mean_cb`, `writer_output_cb`, `batch_var_cb`, `weight_cb`, `bias_cb` | `dfb::src`, `dfb::dst`, `dfb::batch_var`, `dfb::weight`, `dfb::bias` |
| compute CTA 2–11 (`:373-382`) | 10 magic CB indices | `DFBBinding`s (9 unconditional + `writer_output` when typecast, via the `#ifdef`-gated handle alias) |
| writer CTA 0,1 (`:311-312`) | `weight_has_value`, `bias_has_value` — read **only** by `if constexpr` at `writer_batch_norm.cpp:107,119` | promoted to defines `WEIGHT_HAS_VALUE` / `BIAS_HAS_VALUE` |
| reader RTA slots 9,10; writer slots 12,13 | dead RTAs (A1) | **not declared** |

Named CTAs:

- reader: `fill_eps_fp32`
- writer: `batch_stat_is_fp32`, `param_is_fp32`
- compute: `weight_has_value`, `bias_has_value`, `needs_output_typecast`, `tc_in_fmt`, `tc_out_fmt`
  (both compute sources consume `weight_has_value` / `bias_has_value` through **runtime** `if`s, so
  they must stay args, not defines)

Named RTAs:

- reader (8): `eps`, `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`
- writer (7): `start_tile_id`, `num_tiles`, `HtWt`, `n_stride`, `c_stride`, `N`, `C`
- compute (3): `num_tiles`, `tile_freq`, `tile_start`

No CRTAs, no varargs.

---

# Hardware configuration and compiler options

Applies identically to both factories.

- **DM kernels.** Legacy uses bare `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` — both are
  empty structs, i.e. the conventional resolved triples (reader `RISCV_1`/`NOC_0`/`DM_DEDICATED_NOC`,
  writer `RISCV_0`/`NOC_1`/`DM_DEDICATED_NOC`). Ported with the arch-agnostic TTNN helpers:
  `ttnn::create_reader_datamovement_config(device->arch())` and
  `ttnn::create_writer_datamovement_config(device->arch())`. No custom `noc_mode` anywhere, so no
  paired per-node setting to carry.
- **Compute kernels — Style A.** `ttnn::to_compute_hardware_config(device->arch(),
  operation_attributes.compute_kernel_config)`, which maps the same four knobs
  `get_compute_kernel_config_args` returns (`math_fidelity` → `fpu_math_fidelity`;
  `math_approx_mode` → `sfpu_precision_mode`; `fp32_dest_acc_en` → `enable_32_bit_dest`;
  `dst_full_sync_en` → `double_buffer_dest = !dst_full_sync_en`). `get_compute_kernel_config_args` is
  still called for the raw `fp32_dest_acc_en` value, which drives the kernel-source selection and the
  `unpack_modes` table.
- **`bfp_pack_precision_mode`**: legacy leaves `bfp8_pack_precise` at its `false` default in both
  factories ⇒ **do nothing**; the Metal 2.0 default `Precision::Approximate` coincides.
- **`opt_level`**: `grep -n opt_level` over both factories returns nothing — the field is absent from
  every `KernelDescriptor`. Absent on a `ComputeConfigDescriptor` still resolves to **O3**, so each
  compute `KernelSpec` sets `compiler_options.opt_level = KernelBuildOptLevel::O3` **explicitly**.
  The four DM kernels' absent field resolves to `O2`, which is Metal 2.0's default — nothing to do.

## `unpack_modes`

Reindexed from CB id to DFB name and value-translated
(`UnpackToDestFp32` → `UnpackMode::UnpackToDest`; `Default` → `UnpackToSrc`, expressed by **omitting**
the entry). Both factories build the legacy vector only under `fp32_dest_acc_en`, so when that is
false the Metal 2.0 table is **empty**.

- **Unit 1** (`running_statistics_program_factory.cpp:394-411`) — under `fp32_dest_acc_en`, these
  **twelve** DFBs get `UnpackToDest`: `batch_mean`, `batch_var`, `out0`, `old_running_mean`,
  `old_running_var`, `updated_m`, `updated_v`, `momentum`, `one`, `tmp1`, `tmp2`, `tmp3`. All twelve
  are bound on compute in **every** config, so all twelve entries are unconditional.
  `writer_updated_m` / `writer_updated_v` are **not** in the legacy list and must not be added.
- **Unit 2** (`batch_norm_program_factory.cpp:352-368`) — under `fp32_dest_acc_en`, these **eight**:
  `input`, `batch_mean`, `batch_var`, `eps`, `den`, `weight`, `temp_1`, `bias`; **plus** `output_0`
  **only when `needs_output_typecast`** (`:365-367`). `output_0`'s entry is gated on exactly that
  condition, as legacy gates it — an ungated entry would be *legal* here (the DFB is always bound)
  but would silently flip that DFB's unpack mode in the non-typecast config.

Validator interaction, checked against
[`tt_metal/impl/metal2_host_api/program_spec.cpp:921-1073`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp):

- The **required-explicit-entry** rule fires for a *consumed* `Float32` DFB with
  `enable_32_bit_dest = true`. That is exactly the `fp32_dest_acc_en` case, where the legacy vector
  already covers every consumed DFB in both factories — so no entry has to be invented. In both ops
  the only compute-consumed DFBs are ones the legacy list already names (`writer_updated_*` /
  `writer_output` are **producer-only** on compute, and are not `Float32` anyway).
- The **`UnpackToDest` legality** rule (which rejects a ≤16-bit format with `UnpackToDest` on Gen1)
  is short-circuited by `enable_32_bit_dest = true`, which holds whenever we emit an entry. So the
  faithful translation of the legacy vector is accepted even where a listed DFB's format is 16-bit
  (reachable when `fp32_dest_acc_en && !any_float32`).

---

# Applied Patterns

- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — `tmp1`/`tmp2`/`tmp3` on the RunningStatistics compute `KernelSpec`, and `den`/`temp_1` on the
  BatchNorm compute `KernelSpec`: PRODUCER **and** CONSUMER bindings under one shared accessor name.
  Also the typecast-path form of `updated_m`/`updated_v` and `output_0`.
- **Conditional / optional DFB bindings** — applied to the **`TensorParameter`s**, not the DFBs:
  `running_mean` / `running_var` / `weight` / `bias` are declared and bound only when present, with
  `compiler_options.defines` carrying `RUNNING_MEAN_HAS_VALUE` / `RUNNING_VAR_HAS_VALUE` /
  `WEIGHT_HAS_VALUE` / `BIAS_HAS_VALUE` and the kernel `#ifdef`-gating the
  `TensorAccessor(tensor::…)` construction **and** every expression using it. The matching **DFBs**
  go the *other* way and are bound unconditionally — see the endpoint-disposition notes.
  Also applied to the conditionally-declared `writer_updated_m` / `writer_updated_v` /
  `writer_output` DFBs, via the `MEAN_NEEDS_TYPECAST` / `VAR_NEEDS_TYPECAST` /
  `NEEDS_OUTPUT_TYPECAST` defines on the compute kernel.
- **Same-FIFO aliasing (one DFB, multiple kernel-side names)** — the typecast-off configs, where
  legacy points `writer_updated_*_cb` / `writer_output_cb` at the staging CB. Resolved host-side on
  the writer (one accessor name, host picks the DFB) and with a `#ifdef`-gated kernel-side
  `constexpr` handle alias on compute. **Not** `alias_with` — that would model two independent FIFOs
  at one address and lose the pointer coherence the kernels rely on.
- **Pass DFB handles directly to LLKs and kernel-lib helpers** — `dfb::name` flows unchanged into
  `binary_op_init_common`, `unary_op_init_common`, `pack_tile`, `pack_tile_with_dt`, `copy_tile`,
  `add_tiles`/`sub_tiles`/`mul_tiles`, `pack_reconfig_data_format`,
  `copy_tile_to_dst_init_short_with_dt`, the `dest_format_helpers.hpp` `*_to_cb` helpers, and
  `fill_cb_with_value(dfb::one, one_u)`. No `.id` extraction, no temporary `DataflowBuffer` wrappers.
- **Unity-build hygiene for anonymous-namespace symbols** — both factory `.cpp` files land in one
  translation unit and want overlapping names (`BATCH_MEAN_DFB`, `READER`, `OUTPUT_TENSOR`, …), so
  **all** spec-name constants are function-local inside each `create_program_artifacts`.

Not applied: **Aliased DFBs** (`alias_with`) — no legacy CB has multi-element `format_descriptors`.
**Multi-variant factories** — each device-op has a single factory and no variant attribute.
**Removing pybound legacy factory entry points** — no factory entry point is pybound.
**Demoting per-group CTA to RTA** — nothing was ever a per-group CTA.

---

# Deferred / Flagged

New findings from the planning step (none is a stop signal; all are carried into the report):

1. **The brief's CB-index CTA ranges are slightly under-counted for Unit 1's compute kernel.** The
   brief says "compute 2–13" become `dfb::` bindings, but CTAs **14 and 15**
   (`writer_updated_m_cb`, `writer_updated_v_cb`) are CB indices too and likewise become DFB
   bindings — 14 slots, not 12. Verified by reading
   `running_statistics_sfpu_kernel.cpp:61-62` and the host emission at
   `running_statistics_program_factory.cpp:431-432`. Followed the code, not the brief. (Unit 2's
   equivalent, compute CTA 11 = `writer_output_cb`, *is* covered by the brief's "compute 2–11".)
2. **The optional-tensor `*_has_value` CTAs are *writer*-dead post-port.** In both writers the flag
   is read **only** by the `if constexpr` that the port must promote to `#ifdef`, so after promotion
   the CTA has no reader in that kernel and is not declared there. The compute kernels keep theirs
   (runtime `if` in BatchNorm, `if constexpr` + typecast derivation in RunningStatistics). The brief
   lists the flags once, in a combined per-unit list, which reads as "keep them everywhere."
3. **The typecast staging CBs are a Same-FIFO aliasing case, which the brief does not name.** The
   brief describes the config-flip endpoint disposition correctly but does not call out that on the
   typecast-off path `writer_updated_*_cb == updated_*_cb`, i.e. one CB under two kernel-side names.
   That is the part with a genuine trap: modelling it with `advanced_options.alias_with` would be a
   silent correctness bug. Resolution above.
4. **No structural issue the audit missed.** No GlobalCircularBuffer, no `address_offset`, no
   semaphores, no CTA varargs, no offset base pointers, no `TensorAccessor` third argument, no Case 2
   binding, no shared kernel in either direction, no out-of-op call site needing `sem::` or
   `tensor::`. Every gate the audit cleared held up under the inventory.
