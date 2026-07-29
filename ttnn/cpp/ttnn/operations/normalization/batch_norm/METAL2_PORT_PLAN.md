# Port Plan — `normalization/batch_norm`

Port plan for `ttnn/cpp/ttnn/operations/normalization/batch_norm`, ported from `ProgramDescriptor`
(`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

**Audit inputs:** `METAL2_PREPORT_AUDIT.md` (GREEN, every gate cleared) and `METAL2_PORT_BRIEF.md`.

**Two DeviceOperations, one port unit** (per the brief's bundling decision). They share no factory and no
kernel, so the two factories are two independent atomic units; both are converted in this change because
they share a host util, all three donor headers, an identical structural shape, and one user-facing entry
point (`ttnn::batch_norm`, which co-invokes them). Sections below are split per DeviceOperation where the
content differs.

| DeviceOperation | Factory | Kernels |
|---|---|---|
| `BatchNormOperation` | `BatchNormFactory` | `dataflow/reader_batch_norm.cpp` · `dataflow/writer_batch_norm.cpp` · `compute/batch_norm_kernel.cpp` · `compute/batch_norm_sfpu_kernel.cpp` |
| `RunningStatistics` | `RunningStatisticsProgramFactory` | `dataflow/reader_running_statistics.cpp` · `dataflow/writer_running_statistics.cpp` · `compute/running_statistics_kernel.cpp` · `compute/running_statistics_sfpu_kernel.cpp` |

**Each factory runtime-selects its compute kernel *source file*** —
`fmt::format(".../compute/batch_norm_{}.cpp", (fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel")`
(`batch_norm_program_factory.cpp:388-390`, `running_statistics_program_factory.cpp:438-440`). Both sources
per factory are inside the atomic unit and convert in this change. The SFPU variant is the default
(`batch_norm_utils.cpp:31` sets `default_fp32_acc = true`).

---

## Legacy Inventory

### Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** — both DOps. Each exposes a single
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  (`batch_norm_device_operation.hpp:39-42`, `running_statistics_device_operation.hpp:36-39`).
- Variants: **single** per DOp. Each `program_factory_t` is a one-alternative `std::variant`
  (`batch_norm_device_operation.hpp:45`, `running_statistics_device_operation.hpp:42`).
- Custom `compute_program_hash`: **none** — neither device-op defines one; both already use the default
  reflection-based hash. (`RunningStatistics`' was removed earlier, in `975decf0ac2` / #49871.)
  - *Not a custom hash, and not touched:* `BatchNormOperation::operation_attributes_t::to_hash()`
    (`batch_norm_device_operation.cpp:121-123`). The audit and the brief both record that the readiness
    sheet does not score this as a custom hash and that the port must leave it alone.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory
analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Variant: `BatchNormOperation` / `BatchNormFactory`

#### Kernels

Compute-kernel `core_ranges` / CTAs are shared by both selectable sources; the CTA column notes which
indices each source actually reads.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_batch_norm.cpp` | `all_device_cores` (full `compute_with_storage_grid_size`) | `0:input_tensor_cb(c_0)`, `1:eps_cb(c_4)`, `2..:TensorAccessorArgs(input)`, `next:any_float32` (`:303-308`) | none | per-node ×11: `packed_scalar_eps`, `input.buffer()`, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `a_n_stride`, `a_c_stride`, `cN`, `cC`, **`cHt`**, **`cWt`** (`:87-99`) — kernel reads `0..8`; slots 9-10 dead | none | none | `ReaderConfigDescriptor{}` (`:337`) — resolved `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` = reader default |
| writer | `device/kernels/dataflow/writer_batch_norm.cpp` | `all_device_cores` | `0:weight_has_value`, `1:bias_has_value`, `2:batch_mean_tensor_cb(c_1)`, `3:writer_output_cb(c_2 or c_9)`, `4:batch_var_tensor_cb(c_3)`, `5:weight_tensor_cb(c_5)`, `6:bias_tensor_cb(c_6)`, `7..:TensorAccessorArgs(batch_mean, output, batch_var, weight|nullptr, bias|nullptr)`, `next:batch_stat_is_fp32`, `next+1:param_is_fp32` (`:310-328`) | none | per-node ×14: `batch_mean.buffer()`, `batch_var.buffer()`, `weight_arg`, `bias_arg`, `output.buffer()`, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `b_n_stride`, `b_c_stride`, `cN`, `cC`, **`cHt`**, **`cWt`** (`:109-124`) — kernel reads `0..11`; slots 12-13 dead | none | none | `WriterConfigDescriptor{}` (`:346`) — resolved `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = writer default |
| compute | `fmt::format(".../compute/batch_norm_{}.cpp", (fp32_dest_acc_en \|\| any_float32) ? "sfpu_kernel" : "kernel")` (`:388-390`) | `all_device_cores` | 15 slots (`:370-385`): `0:weight_has_value`, `1:bias_has_value`, `2:input_tensor_cb`, `3:batch_mean_tensor_cb`, `4:output_tensor_cb`, `5:batch_var_tensor_cb`, `6:eps_cb`, `7:den_cb`, `8:weight_tensor_cb`, `9:temp_1_cb`, `10:bias_tensor_cb`, `11:writer_output_cb`, `12:needs_output_typecast`, `13:DataFormat::Float32`, `14:tc_out_fmt` — `batch_norm_kernel.cpp` reads `0..10`, `batch_norm_sfpu_kernel.cpp` reads `0..14` | none | per-node ×3: `num_tiles_per_core`, `freq`, `counter` (`:129-130`) — both sources read all three | none | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:394-400`) |

Compute config source: `get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config)`
(`:349-350`) — a resolved TTNN `ComputeKernelConfig` (**Style A**), produced by
`batch_norm::utils::resolve_compute_kernel_config` → `init_device_compute_kernel_config` with
`default_fp32_acc = true`, `default_approx_mode = false`, `default_dst_full_sync_en = false`,
`default_fp32_acc_math_fidelity = HiFi3` (WH) / `HiFi4` (BH) (`batch_norm_utils.cpp:24-37`).
`packer_l1_acc` is unpacked from the tuple and never used. `bfp8_pack_precise` is never set.

`unpack_to_dest_mode` (`:352-368`): `std::vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, Default)`; when
`fp32_dest_acc_en`, slots `{c_0, c_1, c_3, c_4, c_7, c_5, c_8, c_6}` = `UnpackToDestFp32`, **plus `c_2` only
when `needs_output_typecast`** (`:365-367`, landed in #51313). `c_9` deliberately absent.

#### CBs

`total_size = page_size * n`, so `entry_size = page_size` and `num_entries = n` in every row.
`num_tiles_per_cb = 2` and `b_num_tiles_per_cb = 2` (`:191-192`; the latter is initialised from the former
and never reassigned — audit Misc anomaly 6, carried as-is). `format_descriptors[i].tile` is never set on
any descriptor, so `tile_format_metadata` stays `nullopt` throughout.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` `input_tensor_cb` | `a_single_tile_size * 2` | `all_device_cores` | `a_data_format` (input) | `a_single_tile_size` | — |
| `c_1` `batch_mean_tensor_cb` | `b_single_tile_size * 2` | `all_device_cores` | `b_data_format` (batch_mean) | `b_single_tile_size` | — |
| `c_2` `output_tensor_cb` | `(typecast ? interm : c)_single_tile_size * 2` | `all_device_cores` | `typecast ? interm_data_format : c_data_format` | same | — |
| `c_3` `batch_var_tensor_cb` | `d_single_tile_size * 2` | `all_device_cores` | `d_data_format` (batch_var) | `d_single_tile_size` | — |
| `c_4` `eps_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_5` `weight_tensor_cb` | `e_single_tile_size * 2` | `all_device_cores` | `e_data_format` (weight, or `Float16_b` when absent) | `e_single_tile_size` | — |
| `c_6` `bias_tensor_cb` | `f_single_tile_size * 2` | `all_device_cores` | `f_data_format` (bias, or `Float16_b` when absent) | `f_single_tile_size` | — |
| `c_7` `den_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_8` `temp_1_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_9` `writer_cb` *(only `needs_output_typecast`)* | `c_single_tile_size * 2` | `all_device_cores` | `c_data_format` | `c_single_tile_size` | — |

**No GlobalCircularBuffer** anywhere: no `.global_circular_buffer` field, no `remote_cb_config`, no
`experimental::GlobalCircularBuffer`. All 10 are plain `CBDescriptor` + one `CBFormatDescriptor` each — no
multi-element `format_descriptors`, so no aliased CBs. No `CBDescriptor::buffer` / `address_offset`, so no
borrowed-memory CBs.

#### Semaphores

**none** — the op declares no `SemaphoreDescriptor` and contains zero `Semaphore` references.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `batch_norm_program_factory.cpp:307` | `input` | reader RTA 1 (`:90`) → kernel `reader_batch_norm.cpp:38` |
| `:319` | `batch_mean` | writer RTA 0 (`:111`) → `writer_batch_norm.cpp:53` |
| `:320` | `output` (`tensor_return_value`) | writer RTA 4 (`:115`) → `writer_batch_norm.cpp:57` |
| `:321` | `batch_var` | writer RTA 1 (`:112`) → `writer_batch_norm.cpp:61` |
| `:322-323` | `weight` (optional; `nullptr` + RTA `0u` when absent) | writer RTA 2 (`:113`) → `writer_batch_norm.cpp:65` |
| `:324` | `bias` (optional; `nullptr` + RTA `0u` when absent) | writer RTA 3 (`:114`) → `writer_batch_norm.cpp:69` |

All six are the **two-argument** `TensorAccessor(args, addr)` form — no page-size third argument anywhere.
No `ArgConfig::Runtime*` use in any kernel, so no `TensorParameter` relaxation is implicated.

#### Work split

- Driver: `tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, /*row_major=*/true)`
  (`:57`), where `num_output_tiles = output.physical_volume() / tile_hw` (`:44`).
- `num_cores` / `all_cores`: **discarded** (`_unused_num_cores`, `_unused_all_cores`). Kernels are placed on
  `all_device_cores` — the *entire* `compute_with_storage_grid_size` grid — not on the split's `all_cores`.
- `core_group_1`, count `num_tiles_per_core_group_1`; `core_group_2`, count `num_tiles_per_core_group_2`.
- Nodes in **neither** group receive an all-zero RTA block (`:73-79`) and are otherwise idle. This is a
  single-`KernelDescriptor`-per-role split (per-node counts ride as RTAs, not CTAs) — no multiplicity.

### Variant: `RunningStatistics` / `RunningStatisticsProgramFactory`

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_running_statistics.cpp` | `all_device_cores` | `0:batch_mean_tensor_cb(c_0)`, `1:momentum_cb(c_5)`, `2:one_cb(c_6)`, `3..:TensorAccessorArgs(batch_mean)`, `next:any_float32` (`:346-352`) | none | per-node ×11: `packed_scalar_momentum`, `batch_mean.buffer()`, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `a_n_stride`, `a_c_stride`, `cN`, `cC`, **`cHt`**, **`cWt`** (`:85-97`) — kernel reads `0..8`; slots 9-10 dead | none | none | `ReaderConfigDescriptor{}` (`:379`) = reader default |
| writer | `device/kernels/dataflow/writer_running_statistics.cpp` | `all_device_cores` | `0:running_mean_has_value`, `1:running_var_has_value`, `2:batch_var_tensor_cb(c_1)`, `3:output_tensor_cb(c_2)`, `4:old_running_mean_tensor_cb(c_3)`, `5:old_running_var_tensor_cb(c_4)`, `6:writer_updated_m_cb(c_7 or c_12)`, `7:writer_updated_v_cb(c_8 or c_13)`, `8..:TensorAccessorArgs(batch_var, output, running_mean|nullptr, running_var|nullptr)`, `next:old_stat_is_fp32` (`:354-370`) | none | per-node ×13: `batch_var.buffer()`, `running_mean_arg`, `running_var_arg`, `output.buffer()`, `start_tile_id`, `num_tiles_per_core`, `cHtWt`, `b_n_stride`, `b_c_stride`, `cN`, `cC`, **`cHt`**, **`cWt`** (`:107-121`) — kernel reads `0..10`; slots 11-12 dead | none | none | `WriterConfigDescriptor{}` (`:388`) = writer default |
| compute | `fmt::format(".../compute/running_statistics_{}.cpp", (fp32_dest_acc_en \|\| any_float32) ? "sfpu_kernel" : "kernel")` (`:438-440`) | `all_device_cores` | 19 slots (`:416-435`): `0:running_mean_has_value`, `1:running_var_has_value`, `2:batch_mean_tensor_cb`, `3:batch_var_tensor_cb`, `4:output_tensor_cb`, `5:old_running_mean_tensor_cb`, `6:old_running_var_tensor_cb`, `7:updated_m_cb`, `8:updated_v_cb`, `9:momentum_cb`, `10:one_cb`, `11:tmp1_cb`, `12:tmp2_cb`, `13:tmp3_cb`, `14:writer_updated_m_cb`, `15:writer_updated_v_cb`, `16:stat_format_needs_typecast`, `17:DataFormat::Float32`, `18:tc_out_fmt` — `running_statistics_kernel.cpp` reads `0..13`, `running_statistics_sfpu_kernel.cpp` reads `0..18` | none | per-node ×3: `num_tiles_per_core`, `freq`, `counter` (`:126-127`) — **both sources read only slot 0**; slots 1-2 dead | none | none | `ComputeConfigDescriptor{...}` (`:444-450`) |

Compute config source: identical Style A path (`:391-392`), same `batch_norm_utils` defaults (the two DOps
share `resolve_compute_kernel_config`).

`unpack_to_dest_mode` (`:394-411`): when `fp32_dest_acc_en`, slots
`{c_0, c_1, c_2, c_3, c_4, c_7, c_8, c_5, c_6, c_9, c_10, c_11}` = `UnpackToDestFp32`. `c_12` / `c_13`
deliberately absent. **No conditional entry** on this factory (contrast the batch-norm `c_2` entry).

#### CBs

Same shape as above: `entry_size = page_size`, `num_entries = 2`, `tile` never set.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` `batch_mean_tensor_cb` | `a_single_tile_size * 2` | `all_device_cores` | `a_data_format` (batch_mean) | `a_single_tile_size` | — |
| `c_1` `batch_var_tensor_cb` | `b_single_tile_size * 2` | `all_device_cores` | `b_data_format` (batch_var) | `b_single_tile_size` | — |
| `c_2` `output_tensor_cb` | `c_single_tile_size * 2` | `all_device_cores` | `c_data_format` (output) | `c_single_tile_size` | — |
| `c_3` `old_running_mean_tensor_cb` | `d_single_tile_size * 2` | `all_device_cores` | `d_data_format` (running_mean, or `Float16_b` when absent) | `d_single_tile_size` | — |
| `c_4` `old_running_var_tensor_cb` | `e_single_tile_size * 2` | `all_device_cores` | `e_data_format` (running_var, or `Float16_b` when absent) | `e_single_tile_size` | — |
| `c_5` `momentum_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_6` `one_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_7` `updated_m_cb` | `(mean_typecast ? interm : d)_single_tile_size * 2` | `all_device_cores` | `mean_typecast ? interm_data_format : d_data_format` | same | — |
| `c_8` `updated_v_cb` | `(var_typecast ? interm : e)_single_tile_size * 2` | `all_device_cores` | `var_typecast ? interm_data_format : e_data_format` | same | — |
| `c_9` `tmp1_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_10` `tmp2_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_11` `tmp3_cb` | `interm_single_tile_size * 2` | `all_device_cores` | `interm_data_format` | `interm_single_tile_size` | — |
| `c_12` `wm_cb` *(only `needs_mean_typecast`)* | `d_single_tile_size * 2` | `all_device_cores` | `d_data_format` | `d_single_tile_size` | — |
| `c_13` `wv_cb` *(only `needs_var_typecast`)* | `e_single_tile_size * 2` | `all_device_cores` | `e_data_format` | `e_single_tile_size` | — |

No GlobalCircularBuffer, no aliased CB, no borrowed-memory CB — same as above.

#### Semaphores

**none.**

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `running_statistics_program_factory.cpp:351` | `batch_mean` | reader RTA 1 (`:88`) → `reader_running_statistics.cpp:39` |
| `:364` | `batch_var` | writer RTA 0 (`:109`) → `writer_running_statistics.cpp:52` |
| `:365` | `output` (`tensor_return_value`) | writer RTA 3 (`:112`) → `writer_running_statistics.cpp:55` |
| `:366-367` | `running_mean` (optional; **read-modify-write in place**) | writer RTA 1 (`:110`) → `writer_running_statistics.cpp:58` |
| `:368-369` | `running_var` (optional; **read-modify-write in place**) | writer RTA 2 (`:111`) → `writer_running_statistics.cpp:61` |

All five are the two-argument form. `running_mean` / `running_var` are read at
`writer_running_statistics.cpp:86-99` / `:113-128` and written back through the **same** accessor at
`:102-110` / `:130-139` — one `TensorParameter` each, covering both directions.

#### Work split

Identical driver and shape to `BatchNormFactory` (`:57`, `:60-79`), with `num_output_tiles` from the
running-statistics output tensor (`:44`). Same discard of `num_cores` / `all_cores`, same
`all_device_cores` placement, same all-zero RTA block on idle nodes. No multiplicity.

### Shared kernels

**none.** The op owns all 8 kernel sources and no factory outside this directory binds any of them.

Census run per kernel filename (`grep -rl <filename> ttnn/cpp/ttnn/operations/`): the only out-of-directory
hits are in `ttnn/ttnn.egg-info/SOURCES.txt`, a build artifact the disambiguation rule discards. No
`_metal2` sibling exists next to any of the 8 (locational `ls` check on
`device/kernels/{dataflow,compute}/`), so there is **no fork to reuse and none to create**, and no sunset
list. The two factories share no kernel with each other, so the intra-op rung does not apply either.

Three **donor headers** are consumed by `#include` (a function-call escape, not file-path sharing). None
needs an edit, because every function this op calls takes either a bare `uint32_t l1_write_ptr` or a bare
`uint32_t cb_id` — both shapes the Metal 2.0 binding tokens cross unchanged (`dfb::name` has a `constexpr
operator uint32_t()`):

| Donor | Consumed by | Consumed functions | Why no edit |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` | all 4 dataflow kernels | `fill_with_val_bfloat16`, `fill_with_val<Elems,ScalarT>`, `fill_tile_with_first_element<T>`, `fill_tile_with_first_element_bfloat16` | all take a raw `uint32_t l1_write_ptr`; fed by `dfb.get_write_ptr()`. ~35 co-borrowers — must not be touched. |
| `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` | `reader_running_statistics.cpp` | `fill_cb_with_value(uint32_t cb_id, uint32_t, int32_t)` | `uint32_t cb_id`; pass `dfb::one` directly |
| `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` | `batch_norm_kernel.cpp`, `running_statistics_{,sfpu_}kernel.cpp` | `pack_tile_with_dt`, `add/sub/mul_tiles_init_with_dt`, `ckernel::{add,sub,mul}_tiles_to_cb` | all `uint32_t icb`-shaped |

Both `ttnn/cpp/ttnn/kernel/` donors remain `CircularBuffer`-native *internally* while their callers here
are DFB-native. That is invisible at the `uint32_t` boundary and is not port work (kernel-pool owners').

### Flags

- **No unreferenced kernel files** in the directory: all 8 are bound by one of the two factories.
- **No descriptor type outside the audit's Appendix A scan.** Descriptors used: `CBDescriptor` +
  `CBFormatDescriptor`, `KernelDescriptor`, `ReaderConfigDescriptor` / `WriterConfigDescriptor` /
  `ComputeConfigDescriptor`. No `SemaphoreDescriptor`, no GlobalCircularBuffer, no
  `address_offset`, no CTA varargs. Nothing to stop on.
- **Two dead-RTA findings, one beyond the brief.** The brief lists 8 dead RTAs (`cHt`/`cWt` on all four
  dataflow kernels). Independently confirmed, **plus two more the brief does not list**: the
  running-statistics compute RTAs `freq` (slot 1) and `counter` (slot 2) are pushed
  (`running_statistics_program_factory.cpp:126`) but **neither** compute source reads past slot 0
  (`running_statistics_kernel.cpp:12`, `running_statistics_sfpu_kernel.cpp:45`). Ten dead RTAs total.
  The batch-norm compute kernels *do* read all three. See [Dropped Plumbing](#dropped-plumbing).
- **`fmt::format`-selected compute source on both factories** — recorded above; both sources per factory
  are in the atomic unit.
- **Two compute-kernel DFB handles are chosen by a runtime ternary**
  (`batch_norm_kernel.cpp:31-32`, `batch_norm_sfpu_kernel.cpp:42-43`), inside `batchnorm_bcast_tiles`
  whose `weight_has` / `bias_has` parameters are plain runtime `uint32_t`. Consequence for the port: the
  compute kernel must bind **both** `temp_1` and `output` unconditionally. Carried into the census below.
- **A local compute helper takes `DataflowBuffer&`**: `maybe_typecast_stat`
  (`running_statistics_sfpu_kernel.cpp:15-18`). In-file `ALWI` helper, not a donor — no signature change
  needed (it already takes `DataflowBuffer&`; only its `uint32_t` DFB-id parameters change meaning).

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: **`ProgramSpecFactoryConcept`** (base form — cache hit is
  `UpdateTensorArgs` only), `ttnn/api/ttnn/operation_concepts.hpp:119`.
- **Custom `compute_program_hash`**: **none** — nothing to delete on either DOp.
- **Implementation notes**:
  - **The concept flip is atomic per factory.** `ProgramSpecFactoryConcept` requires
    `!ProgramDescriptorFactoryConcept` and `AllFactoriesValid` permits exactly one concept per factory
    (`operation_concepts.hpp:116-119`, `:176-182`), so `create_descriptor` is **removed** from each factory
    struct in the same change that adds `create_program_artifacts`. Both device-op headers change
    signature (a forced edit, not a freelance one).
  - **No pybind entry point to remove.** `batch_norm_nanobind.cpp:70-84` binds only `&ttnn::batch_norm`;
    neither `create_descriptor` nor any other factory entry point is exposed, so exception 2 of the
    ttnn_factory device-op-class edits does not fire. No factory parameter exists for a pybind hook
    either (exception 3 does not fire).
  - **`<tt-metalium/program_descriptors.hpp>` include drops** from both device-op headers (the
    `ProgramDescriptor` return type is gone).
  - **Op-owned tensors: none.** `BatchNormOperation` accepts a caller-supplied preallocated `output`
    (`batch_norm_device_operation.cpp:113-118`), but that is an ordinary optional output tensor reaching
    the factory through `tensor_return_value`, not an op-owned tensor. `ProgramArtifacts::op_owned_tensors`
    is left defaulted on both factories.
  - **Tensor-arg matching stays strict.** No relaxation on any of the 11 `TensorParameter`s; no
    `ArgConfig::Runtime*` in any kernel.

---

## Planned Spec Shape

Default is 1:1 with legacy throughout. Placement is derived: one `WorkUnitSpec` per factory holding all
three kernels over `all_device_cores`, so every DFB's residency matches its legacy `core_ranges`.

### Variant: `BatchNormFactory`

- **KernelSpecs (3)**: `reader`, `writer`, `compute` — one per legacy `KernelDescriptor`. `compute`'s
  `source` keeps the legacy `fmt::format` selection, and its `compile_time_args` are **sized per selected
  source** (the SFPU source reads 4 named CTAs, the FPU source 2) rather than carrying the union.
- **DataflowBufferSpecs (9, +1 conditional)** — one per legacy `CBDescriptor`:
  `input` (c_0) · `batch_mean` (c_1) · `output` (c_2) · `batch_var` (c_3) · `eps` (c_4) · `weight` (c_5) ·
  `bias` (c_6) · `den` (c_7) · `temp_1` (c_8) · **`output_final` (c_9) only when `needs_output_typecast`**.
  `entry_size` / `num_entries` / `data_format_metadata` copied from the legacy descriptor;
  `tile_format_metadata` left `nullopt` (legacy `tile` unset). No `alias_with`, no `borrowed_from`, no
  `allow_instance_multi_binding`.
- **SemaphoreSpecs**: none.
- **TensorParameters (6, 2 conditional)**: `input` · `batch_mean` · `batch_var` · `output` ·
  **`weight` only when present** · **`bias` only when present**.
- **WorkUnitSpecs (1)**: `{reader, writer, compute}` over `all_device_cores`.
- **Op-owned tensors**: none.

### Variant: `RunningStatisticsProgramFactory`

- **KernelSpecs (3)**: `reader`, `writer`, `compute` — same shape, same per-source CTA sizing (SFPU source
  reads 4 named CTAs, FPU source 2).
- **DataflowBufferSpecs (12, +2 conditional)**:
  `batch_mean` (c_0) · `batch_var` (c_1) · `output` (c_2) · `old_running_mean` (c_3) ·
  `old_running_var` (c_4) · `momentum` (c_5) · `one` (c_6) · `updated_running_mean` (c_7) ·
  `updated_running_var` (c_8) · `tmp1` (c_9) · `tmp2` (c_10) · `tmp3` (c_11) ·
  **`writer_updated_mean` (c_12) only when `needs_mean_typecast`** ·
  **`writer_updated_var` (c_13) only when `needs_var_typecast`**.
- **SemaphoreSpecs**: none.
- **TensorParameters (5, 2 conditional)**: `batch_mean` · `batch_var` · `output` ·
  **`running_mean` only when present** · **`running_var` only when present**. The two optionals are
  **read-modify-write through one parameter each** — not split into an in- and an out-binding.
- **WorkUnitSpecs (1)**: `{reader, writer, compute}` over `all_device_cores`.
- **Op-owned tensors**: none.

### DFB endpoint census (re-derived from the kernel bodies, not transcribed)

R = reader, W = writer, C = compute. Every DFB is allocated over `all_device_cores` and all three kernels
of the owning factory run over that same range, so the census is uniform across nodes. Compute line
citations are the SFPU source; the FPU source's touchers are identical in role.

**`BatchNormFactory`:**

| DFB | Touchers | Disposition | Bindings |
|---|---|---|---|
| `input` | R produces (`reader_batch_norm.cpp:66,69`), C consumes (`batch_norm_sfpu_kernel.cpp:89,114`) | legal 1:1 | R PRODUCER, C CONSUMER |
| `batch_mean` | W produces (`writer_batch_norm.cpp:85,93`), C consumes (`:80,187`) | legal 1:1 | W PRODUCER, C CONSUMER |
| `batch_var` | W produces (`writer_batch_norm.cpp:96,105`), C consumes (`:58,78`) | legal 1:1 | W PRODUCER, C CONSUMER |
| `eps` | R produces (`reader_batch_norm.cpp:46,54`), C consumes (`:235,274`) | legal 1:1 | R PRODUCER, C CONSUMER |
| `weight` *(present)* | W produces (`writer_batch_norm.cpp:108,116`), C consumes (`:83,190`) | legal 1:1 | W PRODUCER, C CONSUMER |
| `weight` *(absent)* | named by both W (`writer_batch_norm.cpp:49,64`) and C (`:49`); no FIFO or pointer access executes | **cosmetic 1P+1C** | W PRODUCER, C CONSUMER *(same bindings — unconditional)* |
| `bias` | symmetric to `weight` | as above | W PRODUCER, C CONSUMER |
| `den` | C only — produces (`:57,77`) and consumes (`:81,188`) | **self-loop** | C PRODUCER + C CONSUMER |
| `temp_1` | C only, reached as `dfb_affine_or_out` / `dfb_scaled_output` / `dfb_tmp_1` (`:42-43,90,115,118,137,141,160`); named-but-unused when weight *and* bias are both absent | **self-loop** | C PRODUCER + C CONSUMER *(unconditional — the runtime ternary can select it on any path)* |
| `output` *(no typecast)* | C produces (`:142,159` via the aliases), W consumes (`writer_batch_norm.cpp:133,137`) | legal 1:1 | C PRODUCER, W CONSUMER |
| `output` *(typecast)* | W is redirected to `output_final`, so **C is the only toucher** — produces (`:142,159`) and consumes (`:164,183`) | **self-loop** | C PRODUCER + C CONSUMER |
| `output_final` *(typecast only)* | C produces (`:166,184`), W consumes (`writer_batch_norm.cpp:133,137`) | legal 1:1 | C PRODUCER, W CONSUMER |

The writer's raw-pointer writes — `fill_tile_with_first_element*(dfb_*.get_write_ptr())` at
`writer_batch_norm.cpp:89,91,101,103,112,114,124,126` — are performed by the *same* kernel that
FIFO-produces that DFB, so they are same-binding peeks and add **no** toucher.

**`RunningStatisticsProgramFactory`:**

| DFB | Touchers | Disposition | Bindings |
|---|---|---|---|
| `batch_mean` | R produces (`reader_running_statistics.cpp:73,76`), C consumes (`running_statistics_sfpu_kernel.cpp:94,196`) | legal 1:1 | R PRODUCER, C CONSUMER |
| `batch_var` | W produces (`writer_running_statistics.cpp:79,82`), C consumes (`:219,237`) | legal 1:1 | W PRODUCER, C CONSUMER |
| `output` | C produces (`:95,294`), W consumes (`writer_running_statistics.cpp:144,148`) | legal 1:1 | C PRODUCER, W CONSUMER |
| `momentum` | R produces (`reader_running_statistics.cpp:59,67`), C consumes (`:86,296`) | legal 1:1 | R PRODUCER, C CONSUMER |
| `one` | R produces inside the donor (`cb_fill_helpers.hpp:20,42`, called at `reader_running_statistics.cpp:56`), C consumes (`:87,297`) | legal 1:1 | R PRODUCER, C CONSUMER |
| `old_running_mean` *(present)* | W produces (`writer_running_statistics.cpp:86,99`), C consumes (`:138,156`) | legal 1:1 | W PRODUCER, C CONSUMER |
| `old_running_mean` *(absent)* | named by W (`:46,57`) and by the SFPU C (`:72`), never accessed | **cosmetic 1P+1C** | W PRODUCER, C CONSUMER *(unconditional)* |
| `old_running_var` | symmetric to `old_running_mean` | as above | W PRODUCER, C CONSUMER |
| `tmp1` | C only (`:99,115,137,157`) | **self-loop** | C PRODUCER + C CONSUMER |
| `tmp2` | C only (`:118,134,160,193`) | **self-loop** | C PRODUCER + C CONSUMER |
| `tmp3` | C only (`:139,154,161,192`) | **self-loop** | C PRODUCER + C CONSUMER |
| `updated_running_mean` *(no mean typecast)* | C produces (`:162,183`), W consumes (`writer_running_statistics.cpp:102,110`) | legal 1:1 | C PRODUCER, W CONSUMER |
| `updated_running_mean` *(mean typecast)* | W is redirected to `writer_updated_mean`, so **C is the only toucher** — produces (`:162,183`), consumes inside `maybe_typecast_stat` (`:20,39`) | **self-loop** | C PRODUCER + C CONSUMER |
| `updated_running_var` | symmetric to `updated_running_mean` (`:264,281`) | legal 1:1 / self-loop | as above |
| `writer_updated_mean` *(mean typecast only)* | C produces (`:22,40`), W consumes (`writer_running_statistics.cpp:102,110`) | legal 1:1 | C PRODUCER, W CONSUMER |
| `writer_updated_var` *(var typecast only)* | symmetric (W `:131,139`) | legal 1:1 | C PRODUCER, W CONSUMER |

**Roll-up — my census agrees with the brief in every row.** legal 1:1 ×13–15 · self-loop ×5–8 ·
cosmetic 1P+1C ×0–4 · **`allow_instance_multi_binding` ×0** · **dead-DFB drop ×0** (counts vary with
config). No DFB reaches ≥3 touchers and none has two kernels locked to the same FIFO role, so the
multi-binding advanced option is never set. Nothing is dropped as dead: every DFB is at minimum *named*
by a kernel, and the ambiguous named-but-unaccessed class is bound cosmetic 1P+1C per the owner's ruling.

**How the two config-selected writer targets are expressed without a kernel-side `#ifdef`:** the writer
binds *one* DFB under the accessor name it already uses (`dst`, `updated_mean`, `updated_var`), and the
host varies only which `DFBSpecName` that binding points at — `needs_output_typecast ? OUTPUT_FINAL :
OUTPUT`, `needs_mean_typecast ? WRITER_UPDATED_MEAN : UPDATED_RUNNING_MEAN`, likewise for var. Legacy did
exactly this by assigning `writer_output_cb` / `writer_updated_m_cb`. The compute kernel, which names
*both* ends on the typecast path, does need the `#ifdef` (see [Applied Patterns](#applied-patterns)).

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Each factory emits exactly one `KernelDescriptor` per
role (reader / writer / compute) over a single `core_ranges` (`all_device_cores`); the per-node tile counts
travel as RTAs, not as per-group CTAs. So there is no legacy two-descriptor split to preserve, and no
same-source `KernelSpec` pair to create. Nothing is demoted from CTA to RTA anywhere in this port.

---

## Dropped Plumbing

### `BatchNormFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `batch_norm_program_factory.cpp:90` (reader RTA 1) | `input_tensor.buffer()` | `TensorBinding{INPUT_T, "input"}` on reader; `TensorAccessor(tensor::input)` |
| `:307` (reader CTA 2..) | `TensorAccessorArgs(input.buffer()).append_to(...)`; kernel `TensorAccessorArgs<2>()` | same `TensorBinding` — host plumbing and kernel offset chain both gone |
| `:304` (reader CTA 0) | `input_tensor_cb` (CB index) | `DFBBinding{INPUT_DFB, "input", PRODUCER}` |
| `:305` (reader CTA 1) | `eps_cb` (CB index) | `DFBBinding{EPS_DFB, "eps", PRODUCER}` |
| `:98-99` (reader RTA 9-10) | `cHt`, `cWt` | **dropped — never read** by `reader_batch_norm.cpp` (reads `0..8`) |
| `:111` (writer RTA 0) | `batch_mean_tensor.buffer()` | `TensorBinding{BATCH_MEAN_T, "batch_mean"}` |
| `:112` (writer RTA 1) | `batch_var_tensor.buffer()` | `TensorBinding{BATCH_VAR_T, "batch_var"}` |
| `:113` (writer RTA 2) | `weight_arg` (`Buffer*`, or literal `0u` when absent) | `TensorBinding{WEIGHT_T, "weight"}` **declared only when present**; absent config declares neither the parameter nor the binding |
| `:114` (writer RTA 3) | `bias_arg` (same shape) | `TensorBinding{BIAS_T, "bias"}`, same conditional treatment |
| `:115` (writer RTA 4) | `c.buffer()` | `TensorBinding{OUTPUT_T, "output"}` |
| `:319-324` (writer CTA 7..) | five `TensorAccessorArgs(...).append_to(...)` incl. two `nullptr` placeholders; kernel's 5-link `next_compile_time_args_offset()` chain (`writer_batch_norm.cpp:37-41`) | the five `TensorBinding`s — the `nullptr` placeholders disappear with the absent parameters |
| `:313-317` (writer CTA 2-6) | `batch_mean_tensor_cb`, `writer_output_cb`, `batch_var_tensor_cb`, `weight_tensor_cb`, `bias_tensor_cb` | five `DFBBinding`s (`batch_mean`/`dst`/`batch_var`/`weight`/`bias`) |
| `:311-312` (writer CTA 0-1) | `weight_has_value`, `bias_has_value` | **promoted to `compiler_options.defines`** `WEIGHT_HAS_VALUE` / `BIAS_HAS_VALUE` — required, because the kernel's `if constexpr` blocks now reference `tensor::weight` / `tensor::bias`, which do not exist in the absent config (see [Applied Patterns](#applied-patterns)). The CTAs are removed, not duplicated. |
| `:123-124` (writer RTA 12-13) | `cHt`, `cWt` | **dropped — never read** by `writer_batch_norm.cpp` (reads `0..11`) |
| `:373-382` (compute CTA 2-11) | ten CB indices | ten `DFBBinding`s on the compute `KernelSpec` |
| `:383` (compute CTA 12) | `needs_output_typecast` | **promoted to define** `NEEDS_OUTPUT_TYPECAST` (SFPU source only) — gates the `dfb::output_final` alias |
| all remaining positional CTAs | positional `std::vector<uint32_t>` | named `compile_time_args`: reader `{fill_eps_fp32}`; writer `{batch_stat_is_fp32, param_is_fp32}`; compute `{weight_has_value, bias_has_value}` (+ `{tc_in_fmt, tc_out_fmt}` on the SFPU source) |
| all positional RTAs | positional per-core `CoreRuntimeArgs` | named `runtime_arg_schema.runtime_arg_names` + `runtime_arg_values` via `AddRuntimeArgsForNode` |

### `RunningStatisticsProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `running_statistics_program_factory.cpp:88` (reader RTA 1) | `batch_mean_tensor.buffer()` | `TensorBinding{BATCH_MEAN_T, "batch_mean"}`; `TensorAccessor(tensor::batch_mean)` |
| `:351` (reader CTA 3..) | `TensorAccessorArgs(batch_mean.buffer())`; kernel `TensorAccessorArgs<3>()` | same `TensorBinding` |
| `:347-349` (reader CTA 0-2) | `batch_mean_tensor_cb`, `momentum_cb`, `one_cb` | three `DFBBinding`s (all PRODUCER on the reader) |
| `:96-97` (reader RTA 9-10) | `cHt`, `cWt` | **dropped — never read** (kernel reads `0..8`) |
| `:109` (writer RTA 0) | `batch_var_tensor.buffer()` | `TensorBinding{BATCH_VAR_T, "batch_var"}` |
| `:110` (writer RTA 1) | `running_mean_arg` (`Buffer*` or `0u`) | `TensorBinding{RUNNING_MEAN_T, "old_running_mean"}` **only when present**; one parameter covers the read *and* the in-place write-back |
| `:111` (writer RTA 2) | `running_var_arg` | `TensorBinding{RUNNING_VAR_T, "old_running_var"}`, same |
| `:112` (writer RTA 3) | `c.buffer()` | `TensorBinding{OUTPUT_T, "output"}` |
| `:364-369` (writer CTA 8..) | four `TensorAccessorArgs(...)` incl. two `nullptr` placeholders; kernel's 4-link offset chain (`writer_running_statistics.cpp:36-39`) | the four `TensorBinding`s |
| `:358-362` (writer CTA 2-7) | `batch_var_tensor_cb`, `output_tensor_cb`, `old_running_mean_tensor_cb`, `old_running_var_tensor_cb`, `writer_updated_m_cb`, `writer_updated_v_cb` | six `DFBBinding`s (`batch_var`/`dst`/`old_running_mean`/`old_running_var`/`updated_mean`/`updated_var`) |
| `:355-356` (writer CTA 0-1) | `running_mean_has_value`, `running_var_has_value` | **promoted to defines** `OLD_RUNNING_MEAN_HAS_VALUE` / `OLD_RUNNING_VAR_HAS_VALUE` — required for the same `tensor::` name-lookup reason |
| `:120-121` (writer RTA 11-12) | `cHt`, `cWt` | **dropped — never read** (kernel reads `0..10`) |
| `:419-432` (compute CTA 2-15) | fourteen CB indices | fourteen `DFBBinding`s |
| `:433` (compute CTA 16) | `stat_format_needs_typecast` | **promoted to defines** `NEEDS_MEAN_TYPECAST` / `NEEDS_VAR_TYPECAST` (SFPU source only) — each already ANDed with the matching `has_value` on both host and kernel today, so the two defines carry the same predicate the kernel derived |
| `:126` (compute RTA 1-2) | `freq`, `counter` | **dropped — neither compute source reads past slot 0.** *Not in the brief's dead-RTA list; found during this inventory.* |
| all remaining positional CTAs | positional vector | named: reader `{fill_momentum_fp32}`; writer `{old_stat_is_fp32}`; compute `{old_running_mean_has_value, old_running_var_has_value}` (+ `{tc_in_fmt, tc_out_fmt}` on the SFPU source) |
| all positional RTAs | positional per-core `CoreRuntimeArgs` | named schema + `AddRuntimeArgsForNode` |

**Semaphore-ID RTAs**: none — the op has no semaphores.
**Page-size 3rd-argument CTAs/RTAs**: none — all 11 `TensorAccessor` constructions are the two-argument
form, so no page-size value is emitted to feed a third constructor argument.
**Case 2 (raw base pointer) bindings**: none — all 11 are Case 1 (access goes through the accessor), so
`get_bank_base_address` is not used and the compute-kernel Case-2 block does not apply.

---

## Applied Patterns

- **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**
  — `den`, `temp_1` (batch-norm) and `tmp1`, `tmp2`, `tmp3` (running-statistics) are compute-only in every
  config: bound PRODUCER **and** CONSUMER on the compute `KernelSpec`, one shared `accessor_name`.
  Additionally `output` (batch-norm) and `updated_running_mean` / `updated_running_var`
  (running-statistics) become compute-only self-loops **on their typecast configs**, where the writer is
  redirected to the `c_9` / `c_12` / `c_13` staging DFB.
- **[Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)**
  — the named-but-unaccessed optional DFBs (`weight`, `bias`, `old_running_mean`, `old_running_var` in
  their absent configs) are assigned writer PRODUCER + compute CONSUMER. The roles are cosmetic on Gen1
  and cost nothing at runtime; nothing is dropped as dead. Owner-confirmed disposition.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)**
  — applied in three places, each with a matching `KernelSpec::compiler_options.defines` entry and a
  kernel-side `#ifdef`:
  1. **Optional tensors.** `weight` / `bias` (batch-norm writer) and `running_mean` / `running_var`
     (running-statistics writer) have no `TensorParameter` at all in the absent config, so
     `tensor::weight` etc. do not exist. The legacy gate was a **CTA** driving `if constexpr`, which still
     name-looks-up the discarded branch — so the gate is **promoted to a define**
     (`WEIGHT_HAS_VALUE`, `BIAS_HAS_VALUE`, `OLD_RUNNING_MEAN_HAS_VALUE`, `OLD_RUNNING_VAR_HAS_VALUE`) and
     the `if constexpr` blocks become `#ifdef` blocks. The accessor *declarations* are gated too.
     *The corresponding DFBs stay unconditionally bound* (cosmetic 1P+1C above) — only the **tensor**
     bindings are conditional, so no `if constexpr` on a DFB name needs promoting.
  2. **`output_final`** (batch-norm compute, `c_9`) is bound only when `needs_output_typecast`; the SFPU
     compute source gates its alias under `NEEDS_OUTPUT_TYPECAST`.
  3. **`writer_updated_mean` / `writer_updated_var`** (running-statistics compute, `c_12` / `c_13`) are
     bound only when `needs_mean_typecast` / `needs_var_typecast`; the SFPU compute source gates each
     alias under `NEEDS_MEAN_TYPECAST` / `NEEDS_VAR_TYPECAST`.
  The `unpack_modes` entries for conditionally-bound DFBs are gated on the *same* predicate as the
  binding (the validator rejects a key naming a DFB the kernel doesn't bind) — this only ever matters for
  batch-norm's `output`, the one conditional entry legacy had.
- **[Same-FIFO aliasing, path-dependent variant](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)**
  — the compute kernels reach one DFB through more than one `uint32_t` name. Each is a **handle alias**,
  never a second `DFBBinding` and never `alias_with`:
  - `auto dfb_bcast = dfb_batch_mean; auto dfb_other = dfb_input;` (both compute sources) — plain aliases.
  - `dfb_affine_or_out` / `dfb_scaled_output` — *runtime* ternaries over `temp_1` / `output`
    (`batch_norm_kernel.cpp:31-32`, `batch_norm_sfpu_kernel.cpp:42-43`). `dfb::name`'s `constexpr operator
    uint32_t()` makes both arms `uint32_t`, so the ternaries port unchanged; both arms stay bound
    unconditionally.
  - the `#else` arms of the three typecast `#ifdef`s above alias the staging DFB, matching legacy's
    `writer_output_cb = output_tensor_cb` / `writer_updated_m_cb = updated_m_cb` assignment exactly. The
    alias is inert on that path (`if constexpr (NeedsTypecast)` elides every use).
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)**
  — every LLK and donor call site (`add_tiles`, `pack_tile`, `copy_tile`, `binary_op_init_common`,
  `unary_op_init_common`, `pack_reconfig_data_format`, `copy_tile_to_dst_init_short_with_dt`,
  `pack_tile_with_dt`, `ckernel::{add,sub,mul}_tiles_to_cb`, `fill_cb_with_value`) receives `dfb::name`
  (or a `uint32_t` alias of it) directly. No `.id` extraction, no temporary `DataflowBuffer` wrapper.
- **[Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)**
  — the two factories live in the same CMake target and both introduce `KernelSpecName` / `DFBSpecName` /
  `TensorParamName` constants with overlapping words (`batch_mean`, `output`, `batch_var`, `reader`, …).
  Each factory keeps its existing `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }` wrapper, which
  the build already makes per-TU unique, and the new constants go inside it.
- **Not applied**: [Aliased DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
  (no legacy multi-`buffer_index` `CBDescriptor`);
  [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  (nothing demoted; there was no per-group CTA to begin with);
  [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  (no borrowed, lent, or intra-op kernel);
  [Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  (zero varargs — every argument in all 8 kernels is a distinct field read at a literal index);
  [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
  (each factory builds one `ProgramSpec`; the compute *source* selection is a path inside it, not a
  variant with its own DFB set);
  [Removing pybound legacy factory entry points](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points)
  (no pybound factory entry point).

## Hardware configuration plan

**Data movement (4 kernel specs).** Both factories' DM configs are default-constructed
`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`batch_norm_program_factory.cpp:337,346`,
`running_statistics_program_factory.cpp:379,388`) — resolved values `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`
and `(RISCV_0, NOC_1, DM_DEDICATED_NOC)`, i.e. exactly the reader and writer defaults. So each takes the
arch-agnostic TTNN helper: `ttnn::create_reader_datamovement_config(device->arch())` /
`create_writer_datamovement_config(device->arch())`. On Gen1 these forward to
`CreateReaderGen1DataMovementConfig()` / `CreateWriterGen1DataMovementConfig()` — the exact functions the
brief names — and they additionally supply the Gen2 branch for free, which is why the recipe's
[Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#data-movement-kernels)
section prefers them for a TTNN port. No custom triple, no `DM_DYNAMIC_NOC`, nothing to replicate by hand.

**Compute (2 kernel specs).** **Style A** — both factories resolve a TTNN `ComputeKernelConfig` and read it
through `get_compute_kernel_config_args` (a pure 5-tuple unpack). So the translation is
`ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`, which
applies all four knobs including the two representation changes:

| legacy `ComputeConfigDescriptor` field | Metal 2.0 `ComputeGen1Config` | transform |
|---|---|---|
| `math_fidelity` | `fpu_math_fidelity` | 1:1 |
| `math_approx_mode` (bool) | `sfpu_precision_mode` (`Precision`) | `false` → `Precise` (the op's default) |
| `fp32_dest_acc_en` | `enable_32_bit_dest` | 1:1 |
| `dst_full_sync_en` | `double_buffer_dest` | **inverted** — `double_buffer_dest = !dst_full_sync_en`; the helper does the inversion |
| *(never set)* `bfp8_pack_precise` | `bfp_pack_precision_mode` | left at the Metal default `Approximate`, which coincides with the legacy default — no action |
| `unpack_to_dest_mode` (vector, CB-id-indexed) | `unpack_modes` (`Table<DFBSpecName, UnpackMode>`) | **re-keyed and re-meant** — set by hand on the returned Gen1 config; see below |

`packer_l1_acc` is unpacked from the legacy tuple and never applied to the descriptor, so it is not
translated (and `ComputeGen1Config` has no such field).

**`unpack_modes` — translated entry by entry, never memcpy'd.** `UnpackToDestFp32` → `UnpackMode::UnpackToDest`;
`Default` → `UnpackToSrc`, expressed by **omitting** the entry. Both factories build the table only when
`fp32_dest_acc_en` (matching the legacy `if`), so when it is false the table is empty — which reproduces
the legacy all-`Default` vector exactly.

- `BatchNormFactory`: `{input, batch_mean, batch_var, eps, den, weight, temp_1, bias}` → `UnpackToDest`,
  **plus `output` only when `needs_output_typecast`** (the conditional entry from #51313, kept conditional
  on the same predicate as the binding). `output_final` gets **no** entry — legacy correctly omitted it
  because the compute kernel only ever packs into it.
- `RunningStatisticsProgramFactory`: `{batch_mean, batch_var, output, old_running_mean, old_running_var,
  updated_running_mean, updated_running_var, momentum, one, tmp1, tmp2, tmp3}` → `UnpackToDest`.
  `writer_updated_mean` / `writer_updated_var` get no entry, as in legacy.

Validator cross-check against `tt_metal/impl/metal2_host_api/program_spec.cpp:921-1072`, done before
writing the code:
- *Required-entry rule* (`enable_32_bit_dest=true` + consumed Float32 DFB ⇒ entry mandatory): satisfied.
  Every DFB either compute source consumes has an entry whenever `fp32_dest_acc_en` is true. The
  config-conditional consumers are covered too — batch-norm `output` is consumed only when
  `needs_output_typecast`, exactly when its entry is added; running-statistics `updated_running_*` are
  consumed only on their typecast paths and always carry an entry.
- *`UnpackToDest` legality*: entries exist only when `enable_32_bit_dest == true`, and the table says
  "UnpackToDest, consumer, enable=true → accepted" on every generation and element width. So the
  `fp32_dest_acc_en=true` + all-bfloat16 case (legal in this op, and reachable) does **not** trip the
  Gen1 "≤16-bit + UnpackToDest" rejection, which fires only when `enable_32_bit_dest` is false.
- *Producer-only entries are inert and tolerated*: running-statistics `output` is a pack destination only,
  yet legacy lists it. The entry is kept for fidelity and the validator explicitly tolerates it.
- *No entry names an unbound DFB*: the conditional entry is gated on the binding's own predicate.

**Gen2 is out of scope.** No `ComputeGen2Config` / `DataMovementGen2Config` is populated and no
`if (arch == QUASAR)` branch is added; the two TTNN helpers supply whatever Gen2 branch exists for the
default cases.

---

## Deferred / Flagged

- **New finding: two more dead RTAs than the brief lists.** `freq` and `counter` are pushed to the
  running-statistics compute kernel (`running_statistics_program_factory.cpp:126`) but neither compute
  source reads past RTA slot 0. Dropped in this port (behaviour-preserving: an unread arg has no effect),
  and reported. Brings the op's dead-RTA total from the brief's 8 to 10. The batch-norm compute kernels do
  read all three, so this asymmetry is real, not a mis-read.
- **`num_reader_args` / `num_writer_args` / `num_kernel_args` constants disappear.** The legacy idle-core
  zero-fill counters (`batch_norm_program_factory.cpp:61-63`,
  `running_statistics_program_factory.cpp:60-62`) encoded the inflated arg counts. Metal 2.0 requires
  every *named* RTA to be set on every node the kernel runs on, so idle nodes now receive the same named
  arguments as working nodes, all valued `0` — the same bytes legacy wrote, reached through the named
  path instead of a separate count-based zero-fill. The counters are removed with the plumbing.
- **Idle-node behaviour is preserved verbatim, including two known latent defects.** With
  `num_tiles == 0` the readers still fill and push the `eps` / `momentum` / `one` DFBs, and still evaluate
  `start_tile_id / (HtWt * C)` with a zero divisor (audit Misc anomaly 5). One deliberate difference with
  no observable effect: an idle node's tensor base addresses now arrive from the `TensorBinding` (real
  addresses) rather than as the literal `0u` legacy pushed. Nothing dereferences them — the read/write
  loops are bounded by `num_tiles == 0` — so the accessors are constructed and never used, exactly as
  before.
- **Three latent defects in the non-default compute path are NOT fixed here** (audit Misc anomalies 1-3,
  all in `running_statistics_kernel.cpp`): a `push_back` on `dfb_out0` with no matching `reserve_back`
  (`:57-59`), a nested `tile_regs_acquire()` bracket (`:40-58`), and an output tile packed from undefined
  DST when both running-stat tensors are absent (`:57`). Ported as-is, byte-for-byte. They route to the
  ops team; see `METAL2_PORT_REPORT.md`.
- **No structural issue the audit missed.** No GlobalCircularBuffer, no `get_cb_tiles_*_ptr`, no cursor
  surgery, no Case 2 binding, no out-of-op kernel edit, no host-computed base+offset. Nothing triggers
  [§When the discipline doesn't fit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#when-the-discipline-doesnt-fit).
