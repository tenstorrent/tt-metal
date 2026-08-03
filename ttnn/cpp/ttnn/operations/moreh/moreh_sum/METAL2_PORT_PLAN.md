# Port Plan — `ttnn/cpp/ttnn/operations/moreh/moreh_sum`

Port plan for `moreh_sum`, ported from the legacy `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass:** all **six** factories and all **16** kernels, converted together.
The six factories are structurally parallel (three reduced-dim shapes × float/int32), the
16 kernels are owned exclusively by this op, and two dataflow kernels are shared *between*
two of this op's own factories — so a factory-at-a-time split would have forced an
intra-op `_metal2` fork for zero benefit. See *Shared kernels* below.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — six `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  at `device/moreh_sum_device_operation.hpp:34,41,48,55,62,69`
- Variants: **six factories** in one `program_factory_t` variant
  (`moreh_sum_device_operation.hpp:75-81`), selected by dtype × reduced-dim at
  `moreh_sum_device_operation.cpp:17-39`: `INT32` → the `*Int` factories, else the float
  ones; `dim == rank-1` → W, `dim == rank-2` → H, else NC.
- Custom `compute_program_hash`: **none** — no `compute_program_hash` anywhere in the op;
  already on the default reflection-based hash. Nothing to delete.

*(Metal 2.0 target concept — `ProgramSpecFactoryConcept` — was chosen by the audit; carried
forward under [TTNN ProgramFactory](#ttnn-programfactory).)*

Facts common to all six factories, recorded once:

- **Work split**: two core groups, two compute `KernelDescriptor`s (group 2 only when
  non-empty), one reader and one writer over `all_cores`.
- **Semaphores**: **none** in any factory — the op has no `SemaphoreDescriptor` at all.
  Synchronization is entirely CB FIFO.
- **`opt_level`**: `grep -n opt_level` over the whole op directory returns **zero** hits in
  code. Every `KernelDescriptor::opt_level` is therefore absent → resolves to legacy `O2`
  on the six readers / six writers, and legacy **`O3`** on the twelve compute descriptors.
- **Tensor accessors**: identical in every factory — `input` on the reader, `output` on the
  writer, both **Case 1** (consumed only through `TensorAccessor`), both arriving as a
  `Buffer*` entry in the RTA list (never `->address()`).
- **Aliased CBs / GlobalCircularBuffer / `address_offset` / borrowed memory**: none. Every
  `CBDescriptor` has a single-element `format_descriptors` and no `buffer` field.
- **Op-owned tensors**: none.
- `source_type = KernelDescriptor::SourceType::FILE_PATH` on every kernel (no inline source).

---

### Variant: `MorehSumHFactory` (`device/moreh_sum_h_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_sum_h_impl_kernels/reader_moreh_sum_h.cpp` | `all_cores` | `Ht`, `Wt`, `HtWt`, then `TensorAccessorArgs(*input.buffer())` (:142) | — | `input_buf`, `col_start_tile_id`, `curr_col_in_batch`, `num_cols_per_core`, `mask_h` (:236-242) | — | `REDUCE_SCALER=1`; `DO_MASK_H=1` iff `do_mask_h` | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_sum_h_impl_kernels/writer_moreh_sum_h.cpp` | `all_cores` | `CBIndex::c_16`, then `TensorAccessorArgs(*output.buffer())` (:159-160) | — | `output_buf`, `num_cols_per_core`, `num_cols_read` (:244-250) | — | — | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_sum_h_impl_kernels/moreh_sum_h.cpp` | `core_group_1` | `Ht`, `num_cols_per_core_group_1`, `1`, `origin_H` (:186-191) | — | — | — | `reduce_op_utils::get_defines(SUM, H)`; `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **`O3`** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` |
| compute_2 | *same source* | `core_group_2` | `Ht`, `num_cols_per_core_group_2`, `1`, `origin_H` (:207-212) | — | — | — | *same* | absent → **`O3`** | *same* |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` input | `2 * src0_single_tile_size` | `all_cores` | input dtype | `src0_single_tile_size` | unset |
| `c_2` scaler | `scaler_single_tile_size` | `all_cores` | `Float16_b` | same | unset |
| `c_3` mask_h | `mask_h_single_tile_size` | `all_cores` | `Float16_b` | same | unset |
| `c_24` accum | `intermed_single_tile_size` | `all_cores` | `Float32` iff `fp32_dest_acc_en` else `Float16_b` | same | unset |
| `c_25` masked_input | `intermed1_single_tile_size` | `all_cores` | `Float16_b` | same | unset |
| `c_16` out | `2 * dst_single_tile_size` | `all_cores` | output dtype | `dst_single_tile_size` | unset |

`unpack_to_dest_mode`: all `Default`, **except** `[c_24] = UnpackToDestFp32` when
`fp32_dest_acc_en` (`moreh_sum_h_program_factory.cpp:177-180`).

#### Work split

- Driver: `split_work_to_cores_wt_core_range(all_core_range, num_cols)` where
  `num_cols = other_dims_product * Wt`
- `core_group_1` / `core_group_2`, counts `num_cols_per_core_group_1` / `_2`
- Core walk: `CoreCoord core = {i / num_cores_y, i % num_cores_y}`

---

### Variant: `MorehSumWFactory` (`device/moreh_sum_w_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_sum_w_impl_kernels/reader_moreh_sum_w.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` **first**, then `packed_scaler_value` (:147-149) | — | `input_buf`, `num_tensor_tiles_per_core`, `num_tiles_read`, `mask_w` (:246-251) | — | `DO_MASK_W=1` iff `do_mask_w` | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_sum_w_impl_kernels/writer_moreh_sum_w.cpp` | `all_cores` | `CBIndex::c_16`, then `TensorAccessorArgs(*output.buffer())` (:166-167) | — | `output_buf`, `n_tiles/Wt`, `start/Wt` (:253-259) | — | — | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_sum_w_impl_kernels/moreh_sum_w.cpp` | `core_group_1` | `num_rows_per_core_group_1`, `Wt`, `1`, `origin_W` (:193-198) | — | — | — | `get_defines(SUM, W)`; `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **`O3`** | as H |
| compute_2 | *same source* | `core_group_2` | `num_rows_per_core_group_2`, `Wt`, `1`, `origin_W` (:214-219) | — | — | — | *same* | absent → **`O3`** | *same* |

Note the reader's CTA order: the `TensorAccessorArgs` block is emitted **before** the
scaler, so the kernel reads the scaler at the constexpr-computed offset
`src_args.next_compile_time_args_offset()` (`reader_moreh_sum_w.cpp:17`).

#### CBs

Same six indices as H, with `c_2` sized `2 * scaler_single_tile_size` (vs 1 tile in H) and
`c_3` named mask_w. `c_24` is `Float32` iff `fp32_dest_acc_en` else `Float16_b`; `c_25` is
always `Float16_b`. `unpack_to_dest_mode[c_24] = UnpackToDestFp32` when `fp32_dest_acc_en`
(`moreh_sum_w_program_factory.cpp:184-187`) — same as H.

#### Work split

`split_work_to_cores_wt_core_range(all_core_range, num_rows)`,
`num_rows = other_dims_product * Ht`; writer tile counts divided by `out_dim_divider = Wt`.

---

### Variant: `MorehSumNCFactory` (`device/moreh_sum_nc_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp` **(shared with NCInt)** | `all_cores` | `TensorAccessorArgs(*input.buffer())` only (:117-118) | — | `input_buf`, `num_reduce_input_tile`, `num_tiles_per_core`, `tile_offset`, `dim`, `reduce_tile_size`, `inner_tile_size` (:203-211) | — | **`USE_FPU=1`** | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_sum_nc_impl_kernels/writer_moreh_sum_nc.cpp` **(shared with NCInt)** | `all_cores` | `TensorAccessorArgs(*output.buffer())` only (:131-132) | — | `output_buf`, `num_tiles_per_core`, `tile_offset` (:213) | — | — | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_sum_nc_impl_kernels/moreh_sum_nc.cpp` | `core_group_1` | `num_cols_per_core_group_1`, `num_reduce_input_tile` (:159) | — | — | — | `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **`O3`** | as H |
| compute_2 | *same source* | `core_group_2` | `num_cols_per_core_group_2`, `num_reduce_input_tile` (:175) | — | — | — | *same* | absent → **`O3`** | *same* |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` input | `2 * tile_size_cb` | `all_cores` | output dtype | `tile_size_cb` | unset |
| `c_1` zero | `1 * tile_size_cb` | `all_cores` | output dtype | `tile_size_cb` | unset |
| **`c_24`** | `1 * intermed_tile_size` | `all_cores` | `Float32` iff fp32 else output dtype | `intermed_tile_size` | unset |
| `c_16` out | `2 * tile_size_cb` | `all_cores` | output dtype | `tile_size_cb` | unset |

`unpack_to_dest_mode`: all `Default` (`:150`) — no `c_24` entry, unlike H/W.

#### Work split

`split_work_to_cores(grid, num_output_tiles)`, `num_output_tiles = output.physical_volume() / TILE_HW`.

---

### Variant: `MorehSumHIntFactory` (`device/moreh_int_sum_h_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_sum_h_impl_kernels/reader_moreh_int_sum_h.cpp` | `all_cores` | `Ht`, `Wt`, then `TensorAccessorArgs(input.buffer())` (:129-130) | — | `input_buf`, `col_start_tile_id`, `curr_col_in_batch`, `num_cols_per_core`, `mask_h` (:224-230) | — | `DO_MASK_H=1` iff `do_mask_h` | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_sum_h_impl_kernels/writer_moreh_int_sum_h.cpp` | `all_cores` | `TensorAccessorArgs(output.buffer())` only (:146-147) | — | `output_buf`, `num_cols_per_core`, `num_cols_read` (:232-238) | — | — | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_sum_h_impl_kernels/moreh_int_sum_h.cpp` | `core_group_1` | `num_cols_per_core_group_1`, `Ht`, `origin_H` (:174-177) | — | — | — | `FP32_DEST_ACC_EN=1` (always — forced) | absent → **`O3`** | as H |
| compute_2 | *same source* | `core_group_2` | `num_cols_per_core_group_2`, `Ht`, `origin_H` (:193-196) | — | — | — | *same* | absent → **`O3`** | *same* |

**`fp32_dest_acc_en` is forced on** at `:54-57` (`log_warning` + override) after the tuple
decomposition, so every downstream use — the define and the compute config — sees `true`
regardless of the caller's setting. Same in the W-int (`:56-59`) and NC-int (`:52-55`) factories.

#### CBs

`c_0` (2 tiles), `c_1` mask (1 tile), `c_24` intermed0 (1 tile), `c_16` out (2 tiles) — all
`cb_data_format` = output dtype (`Int32`), all `cb_tile_size`, `all_cores`, no tile field.
`unpack_to_dest_mode`: all `Default` (`:165`).

#### Work split

`split_work_to_cores(grid, num_cols)`, `num_cols = other_dims_product * Wt`.

---

### Variant: `MorehSumWIntFactory` (`device/moreh_int_sum_w_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_sum_w_impl_kernels/reader_moreh_int_sum_w.cpp` | `all_cores` | `TensorAccessorArgs(input.buffer())` only (:131-132) | — | `input_buf`, `num_tensor_tiles_per_core`, `tile_offset`, `mask_w` (:229-234) | — | `DO_MASK_W=1` iff `do_mask_w` | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_sum_w_impl_kernels/writer_moreh_int_sum_w.cpp` | `all_cores` | `TensorAccessorArgs(output.buffer())` only (:148-149) | — | `output_buf`, `n_tiles/Wt`, `offset/Wt` (:236-242) | — | — | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_sum_w_impl_kernels/moreh_int_sum_w.cpp` | `core_group_1` | `num_rows_per_core_group_1`, `Wt`, `origin_W` (:176-179) | — | — | — | `FP32_DEST_ACC_EN=1` (forced) | absent → **`O3`** | as H |
| compute_2 | *same source* | `core_group_2` | `num_rows_per_core_group_2`, `Wt`, `origin_W` (:195-198) | — | — | — | *same* | absent → **`O3`** | *same* |

#### CBs

Identical shape to HInt: `c_0` (2), `c_1` mask_w (1), `c_24` intermed0 (1), `c_16` (2), all
`Int32` / `cb_tile_size`. `unpack_to_dest_mode` all `Default` (`:167`).

#### Work split

`split_work_to_cores(grid, num_rows)`, `num_rows = other_dims_product * Ht`.

---

### Variant: `MorehSumNCIntFactory` (`device/moreh_int_sum_nc_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp` **(shared with NC)** | `all_cores` | `TensorAccessorArgs(*input.buffer())` only (:106-107) | — | 7 args as NC (:189-197) | — | **none — no `USE_FPU`** | absent → `O2` | `ReaderConfigDescriptor{}` |
| writer | `moreh_sum_nc_impl_kernels/writer_moreh_sum_nc.cpp` **(shared with NC)** | `all_cores` | `TensorAccessorArgs(*output.buffer())` only (:117-118) | — | 3 args as NC (:199) | — | — | absent → `O2` | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_sum_nc_impl_kernels/moreh_int_sum_nc.cpp` | `core_group_1` | `num_cols_per_core_group_1`, `num_reduce_input_tile` (:145) | — | — | — | `FP32_DEST_ACC_EN=1` (forced) | absent → **`O3`** | as H |
| compute_2 | *same source* | `core_group_2` | `num_cols_per_core_group_2`, `num_reduce_input_tile` (:161) | — | — | — | *same* | absent → **`O3`** | *same* |

#### CBs

`c_0` (2 tiles), `c_24` intermed0 (1 tile), `c_16` (2 tiles) — all `Int32` / `cb_tile_size`.
**No `c_1`**: this factory omits both the zero CB and the `USE_FPU` define its float sibling
emits, so the shared reader's `prepare_zero_tile` block is preprocessed away here.
`unpack_to_dest_mode` all `Default` (`:136`).

#### Work split

`split_work_to_cores(grid, num_output_tiles)`.

---

### Shared kernels

`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/` over all 16 filenames: the only
consumers are this op's own factories. No kernel is *borrowed* (every `kernel_source` path
is under `moreh_sum/device/`) and none is *lent* (no other op binds a `moreh_sum` kernel).

Two kernels are **intra-op shared**:

| kernel | bound by | existing `_metal2` fork? | rung |
|---|---|---|---|
| `moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp` | `MorehSumNCFactory` + `MorehSumNCIntFactory` | no | **n/a — both binders convert in this change** |
| `moreh_sum_nc_impl_kernels/writer_moreh_sum_nc.cpp` | `MorehSumNCFactory` + `MorehSumNCIntFactory` | no | **n/a — both binders convert in this change** |

Because this pass converts **all six** factories, the two shared sources have no consumer
left behind, so no `_metal2` fork is created and no pointer comment is added — the
[shared-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
rungs do not fire. The one consequence to respect: the two factories' DFB sets **differ**
(`NCInt` allocates no `c_1`), so the shared reader's zero-tile block must stay
preprocessor-gated on `USE_FPU` and the `zero` DFB must be bound only by the float factory.

Out-of-directory coupling is header-only (function-call escape) into
`ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`,
`ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp` and
`ttnn/cpp/ttnn/kernel_lib/{l1_helpers,reduce_helpers_dataflow,reduce_helpers_compute}.hpp`.
Every donor takes either `DataflowBuffer` by value or a `uint32_t` CB id as an NTTP, both of
which a `dfb::name` token satisfies natively — **zero donor-side edits planned**.

### Flags

- No unreferenced kernel files: all 16 are bound by a factory.
- No descriptor type outside the audit's scan: only `KernelDescriptor` and `CBDescriptor`
  appear (no `SemaphoreDescriptor`, no `WorkloadDescriptor`, no `WorkloadBuffer`).
- Pre-existing dead / vestigial items observed and **preserved verbatim** (they are not port
  work — routed to the report): `HtWt` is read at `reader_moreh_sum_h.cpp:21` but never used
  (the column loop strides by `Wt`), so it becomes a named CTA that the kernel still reads
  and still ignores; `dst1` is unused in `moreh_sum_nc.cpp:21`; `num_tiles` is
  `[[maybe_unused]]` in both int H/W factories.
- `moreh_int_sum_nc.cpp` does not `#include "api/dataflow/dataflow_buffer.h"` although it
  uses `DataflowBuffer` (reaches it transitively through `kernel/compute/moreh_common.hpp`).
  Left as-is; the port adds only `experimental/kernel_args.h` there.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (plain), identical for all
  six factories — one wiring pattern covers the op.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**: the six `create_descriptor` declarations in
  `moreh_sum_device_operation.hpp` become `create_program_artifacts` returning
  `ttnn::device_operation::ProgramArtifacts`; the header's
  `<tt-metalium/program_descriptors.hpp>` include is replaced by `"ttnn/metal_v2_artifacts.hpp"`.
  No other device-op-class edit is forced: `moreh_sum_nanobind.cpp` binds only
  `ttnn::moreh_sum` (no `create_descriptor` exposure), and there is no
  `override_runtime_arguments` / `get_dynamic_runtime_args` to unwind.

## Planned Spec Shape

Default 1:1 with legacy in every factory. Resource names below are the `unique_id`s that
drive the generated `dfb::` / `tensor::` / `args::` tokens; all are declared **function-local**
(the six factory `.cpp` files share a unity-build translation unit, so anonymous-namespace
constants would collide — [unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)).

Common to all six:

- **KernelSpecs**: `reader`, `writer`, `compute_g1`, and `compute_g2` (only when
  `core_group_2` is non-empty) — one per legacy `KernelDescriptor`.
- **WorkUnitSpecs**: `wu_g1 = {READER, WRITER, COMPUTE_G1}` on `core_group_1`;
  `wu_g2 = {READER, WRITER, COMPUTE_G2}` on `core_group_2` when present. Reader and writer
  belong to both, so their derived node set is the legacy `all_cores`.
- **TensorParameters**: `input` (bound on the reader) and `output` (bound on the writer),
  both Case 1.
- **SemaphoreSpecs**: none.
- **Op-owned tensors**: none.

| factory | DataflowBufferSpecs (legacy index → name) |
|---|---|
| `MorehSumHFactory` | `c_0`→`input`, `c_2`→`scaler`, `c_3`→`mask_h`, `c_24`→`accum_dst`, `c_25`→`masked_input`, `c_16`→`out` |
| `MorehSumWFactory` | `c_0`→`input`, `c_2`→`scaler`, `c_3`→`mask_w`, `c_24`→`accum_dst`, `c_25`→`masked_input`, `c_16`→`out` |
| `MorehSumNCFactory` | `c_0`→`input`, `c_1`→`zero`, `c_16`→`out` — **`c_24` dropped (dead)** |
| `MorehSumHIntFactory` | `c_0`→`input`, `c_1`→`mask_h`, `c_24`→`intermed0`, `c_16`→`out` |
| `MorehSumWIntFactory` | `c_0`→`input`, `c_1`→`mask_w`, `c_24`→`intermed0`, `c_16`→`out` |
| `MorehSumNCIntFactory` | `c_0`→`input`, `c_24`→`intermed0`, `c_16`→`out` |

`entry_size` / `num_entries` come from the legacy `page_size` and `total_size / page_size`;
`data_format_metadata` from the legacy `data_format`. No legacy CB set
`format_descriptors[i].tile`, so `tile_format_metadata` is left unset everywhere.

### Endpoint census → disposition (re-derived, not transcribed)

Compute is instantiated twice per factory over **disjoint** core groups, so every node sees
exactly one compute instance: this is the disjoint-node work split, **not** the
dual-instance shape, and no 1P+1C assignment question arises from it.

| factory | DFB | config | census (distinct touchers on a node) | disposition |
|---|---|---|---|---|
| H | `input` | all | reader P, compute C | 1P+1C |
| H | `scaler` | all | reader P, compute C | 1P+1C |
| H | `mask_h` | `do_mask_h` | reader P (`#ifdef DO_MASK_H`), compute C | 1P+1C |
| H | `mask_h` | `!do_mask_h` | compute only (reader ref preprocessed away) | **self-loop** |
| H | `accum_dst` | all | compute only (P via `reduce` output, C via `Accumulate::at`) | **self-loop** |
| H | `masked_input` | all | compute only (P `moreh_sum_h.cpp:67`, C via `reduce` input) | **self-loop** (see decision below) |
| H | `out` | all | compute P, writer C | 1P+1C |
| W | `input`, `scaler`, `out` | all | 1P+1C | 1P+1C |
| W | `mask_w` | `do_mask_w` | reader P, compute C | 1P+1C |
| W | `mask_w` | `!do_mask_w` | compute only (plain `if` → still compiled) | **self-loop** |
| W | `accum_dst` | all | compute only (P :61/:68, C :103/:126) | **self-loop** |
| W | `masked_input` | all | compute only (P :84/:91, C :98/:124) | **self-loop** |
| NC | `input`, `zero`, `out` | all | 1P+1C (`zero`: reader P, compute C) | 1P+1C |
| NC | `c_24` | all | **0 touchers** | **dropped** |
| HInt | `input`, `out` | all | 1P+1C | 1P+1C |
| HInt | `mask_h` | `do_mask_h` / `!do_mask_h` | reader P + compute C / compute only | 1P+1C / **self-loop** |
| HInt | `intermed0` | all | compute only (P :63/:81, C :75/:87) | **self-loop** |
| WInt | `input`, `out` | all | 1P+1C | 1P+1C |
| WInt | `mask_w` | `do_mask_w` / `!do_mask_w` | reader P + compute C / compute only | 1P+1C / **self-loop** |
| WInt | `intermed0` | all | compute only (P :62/:79, C :73/:85) | **self-loop** |
| NCInt | `input`, `out` | all | 1P+1C | 1P+1C |
| NCInt | `intermed0` | all | compute only (P via `DataflowBuffer(cb_out)` :40, C :32) | **self-loop** |

**Multi-binding flag: nowhere.** No DFB on any node reaches ≥3 distinct touchers or has two
kernels locked to the same FIFO role. My census agrees with the brief on every row. The
in-place int32 sub-tile folds at `writer_moreh_int_sum_h.cpp:32-40` /
`writer_moreh_int_sum_w.cpp:31-39` are raw peeks *by the kernel that already holds `out`'s
CONSUMER binding* — one toucher, not two.

**Two resolutions supplied by the ops team (both applied as directed):**

1. **NC `c_24` — confirmed dead, dropped.** No spec entry; the allocation at
   `moreh_sum_nc_program_factory.cpp:95-103` and the now-unused locals `intermed0_t` (:60),
   `intermed_data_format` (:70) and `intermed_tile_size` (:72) go with it.
2. **H `c_25` (`masked_input`) under `!do_mask_h` — keep the buffer.** The DFB spec is
   declared **unconditionally**; the `if constexpr (do_mask_h)` guard at `moreh_sum_h.cpp:54`
   is left exactly as it is; the object construction stays at `moreh_sum_h.cpp:23`, above the
   `constexpr bool do_mask_h` gate on line 26 (*not* moved inside the guard); and the compute
   kernel binds `masked_input` as **both PRODUCER and CONSUMER in every configuration**. This
   mirrors the merged `moreh_mean` port
   (`moreh_mean/device/kernels/moreh_mean_h.cpp:26` + its H factory's unconditional
   `MASKED_INPUT_DFB` P+C pair). Consequence: because the object is constructed
   unconditionally, `dfb::masked_input` must exist in both configurations — which the
   unconditional binding provides — so no `#ifdef` gate is introduced for it.

## Preserved Multiplicity

Identical in all six factories:

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_1` (`core_group_1`) + `compute_desc_2` (`core_group_2`), same source, per-group CTA differing on one value | `COMPUTE_G1` + `COMPUTE_G2`, same source, same per-group CTA value reproduced | `wu_g1` + `wu_g2` | every compute-bound DFB of that factory: each `KernelSpec` binds the *same* role as its sibling (`input` CONSUMER, `out` PRODUCER, the self-loops P+C). Legal without any flag because the two node sets are **disjoint** |

The per-group CTA (`num_cols_per_core_group_N` / `num_rows_per_core_group_N` /
`units_per_core`) stays a **CTA** on both specs — no demotion to RTA.

## Dropped Plumbing

Per factory. "→ `TensorBinding`" means both the host-side `TensorAccessorArgs(...).append_to(...)`
CTA block and the `Buffer*` RTA entry disappear together, replaced by one `TensorParameter`
plus a `TensorBinding` on the kernel that reads it.

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `moreh_sum_h_program_factory.cpp:142` + `:238` | `TensorAccessorArgs(*input.buffer()).append_to(reader_cta)` + `input_buf` RTA slot 0 | `TensorParameter{input}` + `TensorBinding` on reader (`tensor::src`) |
| `moreh_sum_h_program_factory.cpp:159` | `writer_cta = {static_cast<uint32_t>(CBIndex::c_16)}` — magic CB index | `DFBBinding{OUT_DFB, "out", CONSUMER}` |
| `moreh_sum_h_program_factory.cpp:160` + `:247` | `TensorAccessorArgs(*output.buffer())` + `output_buf` RTA slot 0 | `TensorParameter{output}` + `TensorBinding` on writer (`tensor::dst`) |
| `reader_moreh_sum_h.cpp:12` / `:30` | `src_addr` RTA + `TensorAccessorArgs<3>()` | `TensorAccessor(tensor::src)` |
| `reader_moreh_sum_h.cpp:23,29,35` | `cb_id_in0 = 0`, `cb_id_in2 = 2`, `cb_id_mask_h = 3` — hardcoded CB indices | `dfb::input`, `dfb::scaler`, `dfb::mask_h` |
| `reader_moreh_sum_h.cpp:19-21` | positional CTAs `Ht`, `Wt`, `HtWt` | named CTAs `Ht`, `Wt`, `HtWt` |
| `writer_moreh_sum_h.cpp:11,15,20-21` | `dst_addr` RTA, `cb_id_out` CTA, `TensorAccessorArgs<1>()` | `TensorAccessor(tensor::dst)`, `dfb::out` |
| `moreh_sum_h.cpp:10-13` | positional CTAs | named CTAs `Ht`, `units_per_core`, `NC`, `origin_H` |
| `moreh_sum_h.cpp:15-24` | `tt::CBIndex::c_0/c_2/c_3/c_24/c_25/c_16` constants | `dfb::input/scaler/mask_h/accum_dst/masked_input/out` |
| `moreh_sum_w_program_factory.cpp:148` + `:248` | `TensorAccessorArgs(*input.buffer())` + `input_buf` RTA | `TensorParameter{input}` + reader `TensorBinding` |
| `moreh_sum_w_program_factory.cpp:149` | `packed_scaler_value` positional CTA (at the accessor-computed offset) | named CTA `scaler` |
| `moreh_sum_w_program_factory.cpp:166-167` + `:256` | `c_16` magic index + `TensorAccessorArgs(*output.buffer())` + `output_buf` RTA | `DFBBinding{OUT_DFB,…}` + writer `TensorBinding` |
| `reader_moreh_sum_w.cpp:11-17` | `src_addr` RTA, `TensorAccessorArgs<0>()`, `next_compile_time_args_offset()` chain | `TensorAccessor(tensor::src)`, named CTA `scaler` |
| `reader_moreh_sum_w.cpp:19,23,29` | `cb_id_in2 = 2`, `cb_id_mask_w = 3`, `cb_id_in0 = 0` | `dfb::scaler`, `dfb::mask_w`, `dfb::input` |
| `writer_moreh_sum_w.cpp:11,15,20` | `dst_addr` RTA, `cb_id_out` CTA, `TensorAccessorArgs<1>()` | `TensorAccessor(tensor::dst)`, `dfb::out` |
| `moreh_sum_w.cpp:10-25` | positional CTAs + `tt::CBIndex::*` constants | named CTAs + `dfb::*` |
| `moreh_sum_nc_program_factory.cpp:118` + `:205` | `TensorAccessorArgs(*input.buffer())` + `input_buf` RTA | `TensorParameter{input}` + reader `TensorBinding` |
| `moreh_sum_nc_program_factory.cpp:132` + `:213` | `TensorAccessorArgs(*output.buffer())` + `output_buf` RTA | `TensorParameter{output}` + writer `TensorBinding` |
| `moreh_sum_nc_program_factory.cpp:95-103` | **dead `c_24` `CBDescriptor`** (+ locals :60/:70/:72) | **dropped — no spec** |
| `reader_moreh_sum_nc.cpp:17,21` | `TensorAccessorArgs<0>()` + `ArgFetcher` address read | `TensorAccessor(tensor::input)` |
| `reader_moreh_sum_nc.cpp:22-27` | six `ArgFetcher::get_next_arg_val` reads (fixed run) | six named RTAs |
| `reader_moreh_sum_nc.cpp:30,33` | `cb_id_in0 = 0`, `cb_id_in1 = 1` | `dfb::input`, `dfb::zero` |
| `writer_moreh_sum_nc.cpp:12,16` | `TensorAccessorArgs<0>()` + `ArgFetcher` address read | `TensorAccessor(tensor::output)` |
| `writer_moreh_sum_nc.cpp:17-18,20` | two `ArgFetcher` reads, `cb_id_out = 16` | named RTAs `num_tiles`, `start_id`; `dfb::out` |
| `moreh_sum_nc.cpp:10-18` | positional CTAs + `c_0`/`c_1`/`c_16` constants | named CTAs + `dfb::input`/`zero`/`out` |
| `moreh_int_sum_h_program_factory.cpp:130` + `:226` | `TensorAccessorArgs(input.buffer())` + `input_buf` RTA | `TensorParameter{input}` + reader `TensorBinding` |
| `moreh_int_sum_h_program_factory.cpp:147` + `:235` | `TensorAccessorArgs(output.buffer())` + `output_buf` RTA | `TensorParameter{output}` + writer `TensorBinding` |
| `reader_moreh_int_sum_h.cpp:11-15,22,28` | positional CTAs, `TensorAccessorArgs<2>()`, `src_addr`, `cb_id_in0`, `cb_id_mask_h` | named CTAs, `TensorAccessor(tensor::src)`, `dfb::input`, `dfb::mask_h` |
| `writer_moreh_int_sum_h.cpp:11,13,17` | `TensorAccessorArgs<0>()`, `dst_addr`, `cb_id_out = 16` | `TensorAccessor(tensor::dst)`, `dfb::out` |
| `moreh_int_sum_h.cpp:10-21` | positional CTAs + `c_0`/`c_1`/`c_16`/`c_24` constants | named CTAs + `dfb::input`/`mask_h`/`out`/`intermed0` |
| `moreh_int_sum_w_program_factory.cpp:132` + `:231`, `:149` + `:239` | both accessor CTA blocks + both `Buffer*` RTAs | two `TensorParameter`s + two `TensorBinding`s |
| `reader_moreh_int_sum_w.cpp:11-19` | `TensorAccessorArgs<0>()`, `src_addr`, `cb_id_in0`, `cb_id_mask_w` | `TensorAccessor(tensor::src)`, `dfb::input`, `dfb::mask_w` |
| `writer_moreh_int_sum_w.cpp:11,13,17` | `TensorAccessorArgs<0>()`, `dst_addr`, `cb_id_out = 16` | `TensorAccessor(tensor::dst)`, `dfb::out` |
| `moreh_int_sum_w.cpp:10-21` | positional CTAs + `tt::CBIndex::*` constants | named CTAs + `dfb::*` |
| `moreh_int_sum_nc_program_factory.cpp:107` + `:191`, `:118` + `:199` | both accessor CTA blocks + both `Buffer*` RTAs | two `TensorParameter`s + two `TensorBinding`s |
| `moreh_int_sum_nc.cpp:10-18` | positional CTAs + `c_0`/`c_16`/`c_24` constants | named CTAs + `dfb::input`/`out`/`intermed0` |

No semaphore-ID RTAs (no semaphores), no page-size third-argument CTAs (all 10 accessor
sites are 2-arg), no `->address()` folds.

**Varargs: none.** Every RTA in the op is a distinct field read once. `ArgFetcher`
(`kernel/dataflow/moreh_common.hpp:44-53`) is a running `arg_idx++` counter over a **fixed**
run of reads at the top of `reader_moreh_sum_nc.cpp` (7) and `writer_moreh_sum_nc.cpp` (3) —
the recipe's explicit non-signal — so all become named RTAs and both `ArgFetcher` locals go.

## Applied Patterns

- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — the compute accumulators (`accum_dst` in H/W, `intermed0` in the three int factories) and
  `masked_input` in H/W: one kernel bound both PRODUCER and CONSUMER under one accessor name.
- **[Sync-free / single-ended CB → self-loop](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — the mask DFBs under `!do_mask_*`: the reader's production is preprocessed away, leaving
  compute the single toucher, so compute takes the extra PRODUCER binding in that config only.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — `mask_h` / `mask_w` on the **reader**, bound only when `do_mask_*`, matching the
  pre-existing `#ifdef DO_MASK_H` / `DO_MASK_W` gate (already a preprocessor gate in the
  legacy kernels — **no `if constexpr`→`#ifdef` promotion needed**); and `zero` on the shared
  NC reader, bound only by the float NC factory, matching its `#ifdef USE_FPU` gate.
- **[Same-FIFO aliasing / runtime-selected handle](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — `moreh_sum_w.cpp`'s mutable `cb_input` (reassigned `c_0` → `c_25` mid-loop) and
  `moreh_int_sum_nc.cpp:39`'s `cb_out` ternary: both stay **`uint32_t`-valued locals**
  initialised from / assigned the `dfb::` tokens, which convert implicitly at compile time.
  Both candidate DFBs are bound to the kernel in every configuration, so both tokens exist.
  No second `DFBBinding`, no `.id` extraction.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — `dfb::name` flows unchanged into `binary_op_init_common`, `unary_op_init_common`,
  `copy_tile`, `pack_tile`, `matmul_init`/`matmul_tiles`, `add_tiles*`,
  `reconfig_data_format*`, `pack_reconfig_data_format`, `copy_tile_to_dst_init_short`, and as
  an NTTP into `compute_kernel_lib::reduce<>`,
  `dataflow_kernel_lib::prepare_zero_tile<>` / `calculate_and_prepare_reduce_scaler<>`.
- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  — all six factories declare their spec-name constants **function-local**, since the six
  `.cpp` files share one unity-build TU.
- **DFB metadata via the object** (whitelist rule 7) — `get_tile_size(cb_id)` in all six
  readers/writers becomes `dfb_*_obj.get_tile_size()`.

## Deferred / Flagged

- **New finding (planning):** the audit's brief did not flag the **`USE_FPU` asymmetry** on
  the shared NC reader — `MorehSumNCFactory` emits the define and allocates `c_1`, while
  `MorehSumNCIntFactory` emits neither. It is not a new *problem* (the brief does warn that
  the two factories' DFB sets differ), but it is what makes `zero` a *conditional* DFB
  binding rather than a plain one, so it is recorded here as an applied pattern above.
- **New finding (planning):** `moreh_sum_w.cpp:15` declares `auto cb_input = tt::CBIndex::c_0;`
  — deducing the *enum* type, not `uint32_t`. Assigning `dfb::masked_input` to an enum-typed
  variable does not compile, so the local's type must become `uint32_t` for the reassignment
  to stay legal. That is a consequence of the binding-token swap, not a logic change.
- Nothing else surfaced that the audit missed. No stop signal.
