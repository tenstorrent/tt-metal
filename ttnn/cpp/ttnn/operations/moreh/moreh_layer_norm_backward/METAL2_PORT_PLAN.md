# Port Plan — `moreh_layer_norm_backward`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/`, ported from the
`ProgramDescriptor` host API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

The directory holds **two independent device-operations**, each with one program factory. Per invoker
decision D3 both land in one PR, GammaBetaGrad first. Each factory is inventoried and planned separately
below.

---

# Part 1 — `MorehLayerNormBackwardGammaBetaGradOperation`

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` returning `tt::tt_metal::ProgramDescriptor`)
- Variants: single (`program_factory_t = std::variant<MorehLayerNormBackwardGammaBetaGradFactory>`)
- Custom `compute_program_hash`: **none** — no `compute_program_hash`, no `to_hash`, no `attribute_values`
  anywhere in the device-operation. Default reflection-based hash is already in use.

*(Metal 2.0 concept chosen by the audit: `ProgramSpecFactoryConcept` — see the brief's TTNN factory analysis.)*

### Kernels

All paths relative to `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/`.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `reader_moreh_layer_norm_backward_gamma_beta_grad.cpp` | `all_cores` | 0:`gamma_grad_has_value`, 1:`do_mask_h`, then `TensorAccessorArgs` for output_grad / input / mean / rstd | none | per-core: `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf`, `num_cols_per_core`, `num_outer`, `num_inner`, `tile_offset`, `mask_h`, `normalized_dims`, `mean_rstd_height`, `mean_rstd_width` (12) | none | `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **O2** | `ReaderConfigDescriptor{}` |
| writer | `writer_moreh_layer_norm_backward_gamma_beta_grad.cpp` | `all_cores` | 0:`gamma_grad_has_value`, 1:`beta_grad_has_value`, then `TensorAccessorArgs` for gamma_grad / beta_grad | none | per-core: `gamma_grad_buf`, `beta_grad_buf`, `num_cols_per_core`, `tile_offset` (4) | none | none | absent → **O2** | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | `core_group_1` | 0:`num_cols_per_core_group_1`, 1:`origin_H`, 2:`origin_W`, 3:`num_outer`, 4:`num_inner`, 5:`gamma_grad_has_value`, 6:`beta_grad_has_value`, 7:`is_lastdim_layer_norm`, 8:`is_groupnorm` | none | none | none | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_COL`, `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute_2 | same source | `core_group_2` (only when `has_core_group_2`) | as compute_1 but slot 0 = `num_cols_per_core_group_2` | none | none | none | as compute_1 | absent → **O3** | as compute_1 |

`grep -n opt_level` over the op directory returns nothing → every kernel's `opt_level` is the resolved
default (O2 for the DM descriptors, **O3** for the two `ComputeConfigDescriptor`s).

### CBs

`push_cb` (factory `:109-123`) skips zero-tile CBs, so a CB with `num_tiles == 0` is **not allocated**.
`cb_data_format = datatype_to_dataformat_converter(output_grad.dtype())`;
`intermed_cb_format = fp32_dest_acc_en ? Float32 : cb_data_format`.
`page_size == total_size / num_tiles == tile_size(fmt)` for every CB; no `tile` field is ever set.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | role comment |
|---|---|---|---|---|---|---|
| c_0 | 1 × tile | all_cores | cb_data_format | tile_size | — | output_grad(==dy) |
| c_1 | 1 × tile | all_cores | cb_data_format | tile_size | — | input(==x) |
| c_2 | 1 × tile | all_cores | cb_data_format | tile_size | — | mean |
| c_3 | 1 × tile | all_cores | cb_data_format | tile_size | — | rstd |
| c_4 | 1 × tile | all_cores | cb_data_format | tile_size | — | scaler |
| c_5 | `do_mask_h ? 1 : 0` × tile | all_cores | cb_data_format | tile_size | — | mask_h — **conditional** |
| c_16 | 1 × tile | all_cores | cb_data_format | tile_size | — | gamma_grad(==dgamma) |
| c_17 | 1 × tile | all_cores | cb_data_format | tile_size | — | beta_grad(==dbeta) |
| c_24 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | output(==y) |
| c_25 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | y * dy |
| c_26 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | Add[dy] |
| c_27 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | Add[y * dy] |
| c_28 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | x - mean |
| c_29 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | dycopy |

**`c_6` is referenced by the compute kernel but never allocated by the factory** — see Flags.
No `GlobalCircularBuffer` / `global_circular_buffer` / `remote_cb_config` anywhere in the directory.
No aliased CBs (every `format_descriptors` is single-element).

### Semaphores

none — neither factory declares a `SemaphoreDescriptor`.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `…gamma_beta_grad_program_factory.cpp:145` | `tensor_args.output_grad` | reader slot 0 |
| `…gamma_beta_grad_program_factory.cpp:146` | `tensor_args.input` | reader slot 1 |
| `…gamma_beta_grad_program_factory.cpp:147` | `tensor_args.mean` | reader slot 2 |
| `…gamma_beta_grad_program_factory.cpp:148` | `tensor_args.rstd` | reader slot 3 |
| `…gamma_beta_grad_program_factory.cpp:152` | `output_tensor.at(0)` (gamma_grad, optional) | writer slot 0 |
| `…gamma_beta_grad_program_factory.cpp:153` | `output_tensor.at(1)` (beta_grad, optional) | writer slot 1 |

All six are **Case 1** (the kernel builds a `TensorAccessor` and uses page accessors). No raw base
pointers, no host-computed base+offset, no third (page-size) constructor argument.

### Work split
- Driver: `tt::tt_metal::split_work_to_cores(grid, num_inner)` (`:76-78`)
- num_cores / all_cores / core_group_1 / core_group_2 / num_cols_per_core_group_1 / num_cols_per_core_group_2
- Per-core loop covers exactly the assigned cores; anything else `TT_THROW`s (`:263`) — no idle-core padding.

### Shared kernels

none. All three sources live in this op's directory and `grep -rl` over
`ttnn/cpp/ttnn/operations/` finds the InputGrad and GammaBetaGrad factories as the only binders, each
binding its own three/five sources. No `_metal2` fork exists or is needed.

Out-of-directory *function-call* coupling (not shared kernels, crosses cleanly):
`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
(both take `DataflowBuffer` already), and `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
(`compute_kernel_lib::reduce` takes cb ids as `uint32_t` non-type template parameters — `dfb::name`
converts `constexpr`).

### Flags
- **`c_6` (`cb_mask_w`) is referenced by the compute kernel and never allocated.** Dead only because
  `do_mask_w = (origin_W % TILE_W) != 0 && is_groupnorm` and the factory hardwires `is_groupnorm = false`
  (`:50`). Invoker decision D1: `#ifdef`-gate the path, do not allocate, do not enable groupnorm.
- The factory and the kernel derive `do_mask_h` from **different** expressions (factory `:55`
  `… && is_lastdim_layer_norm`; kernel `:64` `… && (is_lastdim_layernorm || is_groupnorm)`). They agree
  only because `is_groupnorm == false`. The DFB binding gates on the **factory's** condition.
- `packer_l1_acc` is destructured from `get_compute_kernel_config_args` and never used (`:81`).
- No unreferenced kernel files in the directory.

---

# Part 2 — `MorehLayerNormBackwardInputGradOperation`

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor`)
- Variants: single (`program_factory_t = std::variant<MorehLayerNormBackwardInputGradFactory>`)
- Custom `compute_program_hash`: **none**

### Runtime kernel-source selection

The factory computes `cb_usage` from the full (small-algorithm) CB footprint and compares it with the
core's available L1 (`:123-128`):

```cpp
const bool use_large_algorithm = cb_usage >= available_L1;
```

Two of three roles switch source on that flag, and the **DFB set switches with them**:

| | small | large |
|---|---|---|
| reader source | `reader_…_input_grad_small.cpp` | `reader_…_input_grad_large.cpp` |
| compute source | `moreh_layer_norm_backward_input_grad_small_kernel.cpp` | `…_large_kernel.cpp` |
| writer source | `writer_moreh_layer_norm_backward_input_grad.cpp` (same) | same |
| c_24 (`dycopy`) entries | `num_inner` | **1** |
| c_25 (`y`) entries | `num_inner` | **1** |
| c_31 entries | 1 | **0 → not allocated** |
| tmp1 / tmp2 / tmp3 | c_29 / c_30 / c_31 | c_28 / c_29 / c_30 |
| `recip_nrstd` | its own CB, c_28 | a kernel-side alias of tmp3 (`…large_kernel.cpp:335`) |

So the port converts **five** sources together: writer + both readers + both compute kernels.

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `reader_…_input_grad_{small,large}.cpp` (runtime-selected) | `all_cores` | 0:`gamma_has_value`, 1:`do_mask_h`, 2:`do_mask_w`, then `TensorAccessorArgs` for output_grad / input / mean / rstd / gamma | none | per-core: `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf`, `gamma_buf`, `num_rows_per_core`, `num_inner`, `tile_offset`, `n_u`, `recip_n_u`, `mask_h`, `mask_w`, `normalized_dims`, `mean_rstd_height`, `mean_rstd_width` (15) | none | `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **O2** | `ReaderConfigDescriptor{}` |
| writer | `writer_moreh_layer_norm_backward_input_grad.cpp` | `all_cores` | `TensorAccessorArgs` for input_grad only | none | per-core: `input_grad_buf`, `num_rows_per_core`, `num_inner`, `tile_offset` (4) | none | none | absent → **O2** | `WriterConfigDescriptor{}` |
| compute_1 | `moreh_layer_norm_backward_input_grad_{small,large}_kernel.cpp` | `core_group_1` | 0:`num_rows_per_core_group_1`, 1:`origin_H`, 2:`origin_W`, 3:`num_inner`, 4:`gamma_has_value`, 5:`is_lastdim_layer_norm`, 6:`is_groupnorm` | none | none | none | `REDUCE_OP=PoolType::AVG`, `REDUCE_DIM=ReduceDim::REDUCE_ROW` if `is_lastdim_layer_norm` else `ReduceDim::REDUCE_SCALAR`, `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | absent → **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute_2 | same source | `core_group_2` (only when `has_core_group_2`) | as compute_1 but slot 0 = `num_rows_per_core_group_2` | none | none | none | as compute_1 | absent → **O3** | as compute_1 |

### CBs

| index | total_size | core_ranges | data_format | page_size | tile | role comment |
|---|---|---|---|---|---|---|
| c_0 | 1 × tile | all_cores | cb_data_format | tile_size | — | output_grad(==dy) |
| c_1 | 1 × tile | all_cores | cb_data_format | tile_size | — | input(==x) |
| c_2 | 1 × tile | all_cores | cb_data_format | tile_size | — | mean |
| c_3 | 1 × tile | all_cores | cb_data_format | tile_size | — | rstd |
| c_4 | 1 × tile | all_cores | cb_data_format | tile_size | — | scaler |
| c_5 | 2 × tile | all_cores | cb_data_format | tile_size | — | n_recip_n |
| c_6 | `gamma_has_value ? 1 : 0` | all_cores | cb_data_format | tile_size | — | gamma — **conditional** |
| c_7 | `(do_mask_h \|\| do_mask_w) ? 2 : 0` | all_cores | cb_data_format | tile_size | — | mask_h_w — **conditional** |
| c_16 | 1 × tile | all_cores | cb_data_format | tile_size | — | input_grad(==dx) |
| c_24 | `im0_t` (small `num_inner`, large 1) | all_cores | intermed_cb_format | tile_size | — | copy output_grad(==dy or dy * gamma) |
| c_25 | `im1_t` (small `num_inner`, large 1) | all_cores | intermed_cb_format | tile_size | — | output(==y) |
| c_26 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | Sum[dy] |
| c_27 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | Sum[y * dy] |
| c_28 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | small: rstd / n · large: tmp1 |
| c_29 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | small: tmp1 · large: tmp2 |
| c_30 | 1 × tile | all_cores | intermed_cb_format | tile_size | — | small: tmp2 · large: tmp3 |
| c_31 | `im7_t` (small 1, large **0 → not allocated**) | all_cores | intermed_cb_format | tile_size | — | small: tmp3 |

No GlobalCircularBuffer, no aliased CBs, no `tile` overrides.

### Semaphores

none.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `…input_grad_program_factory.cpp:180` | `tensor_args.output_grad` | reader slot 0 |
| `…input_grad_program_factory.cpp:181` | `tensor_args.input` | reader slot 1 |
| `…input_grad_program_factory.cpp:182` | `tensor_args.mean` | reader slot 2 |
| `…input_grad_program_factory.cpp:183` | `tensor_args.rstd` | reader slot 3 |
| `…input_grad_program_factory.cpp:184` | `tensor_args.gamma` (optional) | reader slot 4 |
| `…input_grad_program_factory.cpp:187` | `input_grad` (`tensor_return_value`) | writer slot 0 |

All six **Case 1**. No compute kernel constructs a `TensorAccessor`, so the blocked compute-kernel
Case-2 path cannot arise.

### Work split
- Driver: `tt::tt_metal::split_work_to_cores(grid, num_outer)` (`:86-88`)
- Per-core loop covers exactly the assigned cores; `TT_THROW` otherwise (`:305`) — no idle-core padding.

### Shared kernels

none (see Part 1).

### Flags
- `is_groupnorm` is hardwired `false` (`:50`) while all three compute kernels carry live
  `is_groupnorm` branches. Pre-existing; left alone (D1 spirit), routed to the report.
- `packer_l1_acc` destructured and unused (`:91`).
- Two `log_info(tt::LogTest, …)` calls on the algorithm-selection path (`:131`, `:136`) fire on every
  cache miss at the wrong severity. **Deliberately left in place** — the `_large` one is the port's only
  signal that the large path was exercised (see the brief's test gate).
- `output_grad_rank`, `mean_rstd_shape` are computed and (partly) unused. Pre-existing; left alone.
- No unreferenced kernel files.

---

# TTNN ProgramFactory

*Applies to both device-operations.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none (neither device-operation defines one)
- **Op-owned tensors**: none — `ProgramArtifacts::op_owned_tensors` stays defaulted
- **Implementation notes**: the only device-op **header** changes are the factory method signature
  (`static tt::tt_metal::ProgramDescriptor create_descriptor(...)` →
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`) and swapping
  `#include <tt-metalium/program_descriptors.hpp>` for `#include "ttnn/metal_v2_artifacts.hpp"`.
  `program_factory_t` already names a factory struct in both headers, so nothing else moves. No pybind
  cleanup (nothing pybinds `create_descriptor`), no pybind-hook-only factory parameter.

---

# Planned Spec Shape

## GammaBetaGrad

**KernelSpecs** (4, or 3 without core group 2):

| unique_id | source | notes |
|---|---|---|
| `GBG_READER` "reader" | `reader_…_gamma_beta_grad.cpp` | `create_reader_datamovement_config(arch)` |
| `GBG_WRITER` "writer" | `writer_…_gamma_beta_grad.cpp` | `create_writer_datamovement_config(arch)` |
| `GBG_COMPUTE_G1` "compute_g1" | `…_gamma_beta_grad_kernel.cpp` | `to_compute_hardware_config`, `opt_level = O3` |
| `GBG_COMPUTE_G2` "compute_g2" | same source | only when `has_core_group_2`; `opt_level = O3` |

**DataflowBufferSpecs** (14 with mask_h, 13 without) — one per allocated legacy CB, 1:1. No aliasing,
no borrowed memory, no advanced options:

| DFBSpecName const | string | legacy CB | entries | format |
|---|---|---|---|---|
| `DY` | "dy" | c_0 | 1 | cb_data_format |
| `X` | "x" | c_1 | 1 | cb_data_format |
| `MEAN` | "mean" | c_2 | 1 | cb_data_format |
| `RSTD` | "rstd" | c_3 | 1 | cb_data_format |
| `SCALER` | "scaler" | c_4 | 1 | cb_data_format |
| `MASK_H` | "mask_h" | c_5 | 1 | cb_data_format | **only when factory `do_mask_h`** |
| `DGAMMA` | "dgamma" | c_16 | 1 | cb_data_format |
| `DBETA` | "dbeta" | c_17 | 1 | cb_data_format |
| `Y` | "y" | c_24 | 1 | intermed_cb_format |
| `YDY` | "ydy" | c_25 | 1 | intermed_cb_format |
| `DYADD` | "dyadd" | c_26 | 1 | intermed_cb_format |
| `YDYADD` | "ydyadd" | c_27 | 1 | intermed_cb_format |
| `XMM` | "xmm" | c_28 | 1 | intermed_cb_format |
| `DYCOPY` | "dycopy" | c_29 | 1 | intermed_cb_format |

`c_6` gets **no spec** (never allocated; kernel path `#ifdef`-ed out per D1).

**SemaphoreSpecs**: none.

**TensorParameters** (6, or fewer): `OUTPUT_GRAD_T` "output_grad", `INPUT_T` "input", `MEAN_T` "mean",
`RSTD_T` "rstd", `GAMMA_GRAD_T` "gamma_grad" (conditional), `BETA_GRAD_T` "beta_grad" (conditional).

**WorkUnitSpecs** (2, or 1):
```
wu_g1: {GBG_READER, GBG_WRITER, GBG_COMPUTE_G1}  target_nodes = core_group_1
wu_g2: {GBG_READER, GBG_WRITER, GBG_COMPUTE_G2}  target_nodes = core_group_2   (iff has_core_group_2)
```
Reader and writer are members of both, so their derived node set is `core_group_1 ∪ core_group_2 == all_cores`,
matching legacy.

**Op-owned tensors**: none.

### Endpoint census (re-derived from the kernel bodies, not transcribed)

| DFB | touching kernels | roles | disposition |
|---|---|---|---|
| DY | reader (`reserve_back`/`push_back`), compute (`wait_front`/`pop_front`) | 1 locked P, 1 locked C | **1P+1C** |
| X | reader, compute | 1 P, 1 C | **1P+1C** |
| MEAN | reader (`read_mean_rstd`), compute | 1 P, 1 C | **1P+1C** |
| RSTD | reader, compute | 1 P, 1 C | **1P+1C** |
| SCALER | reader (`fill_cb_with_value`), compute | 1 P, 1 C | **1P+1C** |
| MASK_H | reader (`generate_mask_h`), compute | 1 P, 1 C | **1P+1C** (both bindings conditional) |
| DGAMMA | compute (`reserve_back`/`push_back`), writer (`wait_front`/`pop_front`) | 1 P, 1 C | **1P+1C** |
| DBETA | compute, writer | 1 P, 1 C | **1P+1C** |
| Y, YDY, DYADD, YDYADD, XMM, DYCOPY | compute only | 1 toucher, both roles | **self-loop** ×6 |

No DFB has ≥3 touchers or two kernels locked to the same role → **no `allow_instance_multi_binding`
anywhere**. No dead CB among the allocated set. Agrees with the brief's item 8.

## InputGrad

**KernelSpecs** (4, or 3): `IG_READER` "reader", `IG_WRITER` "writer", `IG_COMPUTE_G1` "compute_g1",
`IG_COMPUTE_G2` "compute_g2". Reader and compute sources are picked by `use_large_algorithm`.

**DataflowBufferSpecs** — the list itself branches on `use_large_algorithm`:

| DFBSpecName const | string | small CB | large CB | entries | format |
|---|---|---|---|---|---|
| `DY` | "dy" | c_0 | c_0 | 1 | cb_data_format |
| `X` | "x" | c_1 | c_1 | 1 | cb_data_format |
| `MEAN` | "mean" | c_2 | c_2 | 1 | cb_data_format |
| `RSTD` | "rstd" | c_3 | c_3 | 1 | cb_data_format |
| `SCALER` | "scaler" | c_4 | c_4 | 1 | cb_data_format |
| `N_RECIP_N` | "n_recip_n" | c_5 | c_5 | 2 | cb_data_format |
| `GAMMA` | "gamma" | c_6 | c_6 | 1 | cb_data_format | **only when `gamma_has_value`** |
| `MASK_H_W` | "mask_h_w" | c_7 | c_7 | 2 | cb_data_format | **only when `do_mask_h \|\| do_mask_w`** |
| `DX` | "dx" | c_16 | c_16 | 1 | cb_data_format |
| `DYCOPY` | "dycopy" | c_24 | c_24 | `num_inner` / **1** | intermed_cb_format |
| `Y` | "y" | c_25 | c_25 | `num_inner` / **1** | intermed_cb_format |
| `DYSUM` | "dysum" | c_26 | c_26 | 1 | intermed_cb_format |
| `YDYSUM` | "ydysum" | c_27 | c_27 | 1 | intermed_cb_format |
| `RECIP_NRSTD` | "recip_nrstd" | c_28 | — | 1 | intermed_cb_format | **small only** |
| `TMP1` | "tmp1" | c_29 | c_28 | 1 | intermed_cb_format |
| `TMP2` | "tmp2" | c_30 | c_29 | 1 | intermed_cb_format |
| `TMP3` | "tmp3" | c_31 | c_30 | 1 | intermed_cb_format |

Small allocates 17 DFBs (minus conditionals), large 16 — matching legacy's CB counts and total L1 exactly.
Buffer indices are now framework-assigned, so the legacy c_NN columns are historical only.
No `#ifdef` is needed for the RECIP_NRSTD/TMP3 difference: each compute source has its own `KernelSpec`
bindings, and the large kernel simply never names `dfb::recip_nrstd` (it aliases `dfb::tmp3` instead).

**SemaphoreSpecs**: none.

**TensorParameters** (6, or 5): `OUTPUT_GRAD_T` "output_grad", `INPUT_T` "input", `MEAN_T` "mean",
`RSTD_T` "rstd", `GAMMA_T` "gamma" (conditional), `INPUT_GRAD_T` "input_grad".

**WorkUnitSpecs** (2, or 1): same shape as GammaBetaGrad.

**Op-owned tensors**: none.

### Endpoint census

| DFB | touching kernels | disposition |
|---|---|---|
| DY, X, MEAN, RSTD, SCALER, N_RECIP_N | reader P → compute C | **1P+1C** ×6 |
| GAMMA | reader P → compute C | **1P+1C** (both conditional) |
| MASK_H_W | reader P (`generate_mask_h_w`) → compute C | **1P+1C** (both conditional) |
| DX | compute P → writer C | **1P+1C** |
| DYCOPY, Y, DYSUM, YDYSUM, TMP1, TMP2, TMP3 | compute only | **self-loop** ×7 |
| RECIP_NRSTD (small only) | compute only | **self-loop** |

Small: 8 self-loops; large: 7. No multi-binding flag anywhere, no dead CB. Agrees with brief item 8.

---

# Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| GammaBetaGrad `compute_desc_1` (`core_group_1`) + `compute_desc_2` (`core_group_2`), one source, differing only in CTA slot 0 (`num_cols_per_core_group_{1,2}`) | `GBG_COMPUTE_G1`, `GBG_COMPUTE_G2` | `wu_g1`, `wu_g2` | DY/X/MEAN/RSTD/SCALER/[MASK_H] CONSUMER; DGAMMA/DBETA PRODUCER; Y/YDY/DYADD/YDYADD/XMM/DYCOPY PRODUCER+CONSUMER (self-loop) — each role bound by **both** specs, legal because their node sets are disjoint |
| InputGrad `compute_desc_1` + `compute_desc_2`, one source, differing only in CTA slot 0 (`num_rows_per_core_group_{1,2}`) | `IG_COMPUTE_G1`, `IG_COMPUTE_G2` | `wu_g1`, `wu_g2` | DY/X/MEAN/RSTD/SCALER/N_RECIP_N/[GAMMA]/[MASK_H_W] CONSUMER; DX PRODUCER; the 7–8 intermediates self-looped — again both specs, disjoint node sets |

This is the **disjoint-node** work-split, not the same-grid two-toucher shape: each node runs exactly one
compute instance, so every DFB is an ordinary 1:1 (or self-loop) per node. No multi-binding flag.
Per-group counts stay **CTAs**; demoting them to RTAs is the documented anti-pattern and would cost the
compute kernels their compile-time loop unrolling.

---

# Dropped Plumbing

## GammaBetaGrad

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:145-148` / reader CTA slots 2… | `TensorAccessorArgs(output_grad.buffer()).append_to(...)` ×4 | `TensorParameter` + `TensorBinding` ×4 |
| reader kernel `:124-127` | `TensorAccessorArgs<2>()` chain via `next_compile_time_args_offset()` | gone — `TensorAccessor(tensor::output_grad)` etc. |
| factory `:268-271` / reader RTA slots 0-3 | `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf` (`Buffer*`) | `TensorBinding` (address auto-injected per enqueue) |
| reader kernel `:100-103` | `get_arg_val<uint32_t>(0..3)` address reads | gone |
| reader kernel `:115-120` | `constexpr uint32_t cb_id_* = 0..5` magic indices | `dfb::dy` / `dfb::x` / `dfb::mean` / `dfb::rstd` / `dfb::scaler` / `dfb::mask_h` |
| factory `:144` slot 1 / reader kernel `:123` | CTA `do_mask_h` | promoted to `compiler_options.defines["DO_MASK_H"]` (the CTA's only use was gating the conditional MASK_H binding) |
| factory `:152-153` / writer CTA slots 2… | `TensorAccessorArgs(gamma_grad/beta_grad)` | `TensorParameter` + conditional `TensorBinding` |
| writer kernel `:19-20` | `TensorAccessorArgs<2>()` chain | gone |
| factory `:281` / writer RTA slots 0-1 | `gamma_grad_buf`, `beta_grad_buf` (`Buffer*`, possibly `nullptr`) | conditional `TensorBinding` |
| writer kernel `:11-12` | `get_arg_val<uint32_t>(0..1)` address reads | gone |
| writer kernel `:22-23` | `constexpr uint32_t cb_id_gamma_grad = 16; … = 17;` | `dfb::dgamma`, `dfb::dbeta` |
| factory `:151` slots 0-1 / writer kernel `:17-18` | CTAs `gamma_grad_has_value`, `beta_grad_has_value` | promoted to defines `GAMMA_GRAD_HAS_VALUE` / `BETA_GRAD_HAS_VALUE` (their only use gates the conditional tensor bindings) |
| compute kernel `:20-54` | `constexpr auto cb_* = tt::CBIndex::c_N` ×14 | `dfb::*` binding tokens |
| compute kernel `:32-33` | `cb_mask_w = tt::CBIndex::c_6` — CB never allocated | **no DFB**; path `#ifdef DO_MASK_W`-gated and the define is never emitted (D1) |
| all three kernels | `get_tile_size(cb_id)` free-function calls (5 sites in this factory's kernels) | `dfb_<name>_obj.get_tile_size()` |
| all kernel CTAs | positional `get_compile_time_arg_val(N)` | named `get_arg(args::<name>)` |
| all kernel RTAs | positional `get_arg_val<uint32_t>(N)` | named `get_arg(args::<name>)` |

## InputGrad

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:180-184` | `TensorAccessorArgs(...)` ×5 (output_grad, input, mean, rstd, gamma) | `TensorParameter` + `TensorBinding` (gamma conditional) |
| factory `:187` | `TensorAccessorArgs(input_grad.buffer())` | `TensorParameter` + `TensorBinding` |
| readers `:128-133` (both) | `TensorAccessorArgs<3>()` chain ×5 | gone |
| writer `:16` | `TensorAccessorArgs<0>()` | gone |
| factory `:310-314` / reader RTA slots 0-4 | five `Buffer*` address args | `TensorBinding`s |
| factory `:326` / writer RTA slot 0 | `input_grad_buf` | `TensorBinding` |
| readers `:100-104`, writer `:11` | `get_arg_val<uint32_t>` address reads | gone |
| readers `:116-123`, writer `:18`, computes `:24-60` | `constexpr uint32_t cb_id_* = N` / `tt::CBIndex::c_N` | `dfb::*` binding tokens |
| factory `:179` slot 0 / kernels' `gamma_has_value` CTA | CTA gating `dfb::gamma` / `tensor::gamma` | promoted to define `GAMMA_HAS_VALUE` on reader **and** compute |
| factory `:179` slots 1-2 / readers' `do_mask_h`, `do_mask_w` CTAs | CTAs whose only use is `if (do_mask_h \|\| do_mask_w)` | promoted to define `DO_MASK_H_W` on the reader |
| compute kernels' `dfb::mask_h_w` references | — | gated by define `DO_MASK_H_W`; the kernels' own `do_mask_h` / `do_mask_w` `constexpr` derivations (identical to the factory's) stay and still drive the inner `if`s |
| all five kernels | `get_tile_size(cb_id)` free-function calls (9 sites) | member getter on the object |
| all kernel CTAs / RTAs | positional | named |

Nothing else is dropped. No page-size third accessor argument exists in this op; no semaphore-ID RTA;
no positional CTA survives.

---

# Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — every compute-private intermediate: 6 in GammaBetaGrad, 8 (small) / 7 (large) in InputGrad. One
  accessor name per DFB serving both PRODUCER and CONSUMER, so the kernel builds one object.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — `MASK_H` (GammaBetaGrad), `GAMMA` and `MASK_H_W` (InputGrad), and the conditional `TensorBinding`s
  for `gamma_grad` / `beta_grad` / `gamma`. Host binds conditionally, emits the matching define,
  kernel `#ifdef`-gates. Includes the *promote-a-CTA-gate-to-a-define* sub-case for
  `do_mask_h`, `gamma_grad_has_value`, `beta_grad_has_value`, `gamma_has_value`, `do_mask_h`/`do_mask_w`.
  The `c_6` mask-w path in the GammaBetaGrad compute kernel is the degenerate case: gated by a define
  that is **never** emitted.
- [Same-FIFO aliasing (one DFB, multiple kernel-side names)](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  — the InputGrad compute kernels' per-phase tmp names (`xmm`, `ydy`, `dyadd`, `ydyadd`, `ndy`,
  `ndymdysum`, `yydysum`, `recip_nrstd`, `tmp4`). One `DataflowBufferSpec`, one `DFBBinding`, a
  `constexpr auto` handle alias per phase name, and **one** `DataflowBuffer` object per real DFB with
  the phase-local names becoming `DataflowBuffer&` references to it. Explicitly **not** `alias_with`.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — `copy_tile`, `add_tiles`, `mul_tiles*`, `sub_tiles*`, `binary_op_init_common`, and
  `compute_kernel_lib::reduce<…>` as a non-type template argument.
- [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  — avoided; both factories keep two compute `KernelSpec`s in two `WorkUnitSpec`s.
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  — each factory's spec-name constants live in an anonymous namespace nested inside that factory's own
  named namespace (`…moreh_layer_norm_backward_gamma_beta_grad` vs `…moreh_layer_norm_backward_input_grad`),
  so the two `DY`/`X`/`MEAN`/… sets cannot collide under unity build. The shared
  `make_dfb` / `bind_self_loop` / `unpack_via_src` / `gen1_compute_config` helpers are `inline` in one
  op-local header.

---

# Hardware configuration and `opt_level`

- **Compute — Style A, both factories.** Each factory already re-resolves
  `init_device_compute_kernel_config(arch, operation_attributes.compute_kernel_config)` at its top
  (GammaBetaGrad `:34-35`, InputGrad `:33-34`) and destructures it with
  `get_compute_kernel_config_args`. The port translates that **re-resolved** config with
  `ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config)`, which carries
  `math_fidelity`, the `math_approx_mode` bool → `Precision` mapping, `fp32_dest_acc_en` →
  `enable_32_bit_dest`, and the `dst_full_sync_en` → `double_buffer_dest` **inversion**.
  `bfp_pack_precision_mode` is unset in legacy → left at default.
  *(The brief says only InputGrad re-resolves; GammaBetaGrad does the same thing at `:34-35`. Following
  the source. Noted in the report.)*
- **`unpack_modes`** — legacy sets no `unpack_to_dest_mode`, so every entry translates to
  `UnpackMode::UnpackToSrc`. Metal 2.0 requires an explicit entry for every **consumed Float32** DFB
  when `enable_32_bit_dest` is true, which here is exactly the intermediates (they become Float32
  precisely when `fp32_dest_acc_en`). Hand-listed per DFB via a local `unpack_via_src(cfg, DFB)` helper,
  gated on `fp32_dest_acc_en`:
  - GammaBetaGrad (6): `Y`, `YDY`, `DYADD`, `YDYADD`, `XMM`, `DYCOPY`
  - InputGrad small (8): `DYCOPY`, `Y`, `DYSUM`, `YDYSUM`, `RECIP_NRSTD`, `TMP1`, `TMP2`, `TMP3`
  - InputGrad large (7): `DYCOPY`, `Y`, `DYSUM`, `YDYSUM`, `TMP1`, `TMP2`, `TMP3`

  The io DFBs never need an entry: every io tensor is validated `BFLOAT16`
  (`check_tensor`'s default `data_types` list), so `cb_data_format` is never Float32.
  **No auto-fill helper** — per the invoker's `unpack_modes` policy.
- **Data movement** — both factories use bare `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`,
  i.e. the resolved reader / writer defaults, so the port uses
  `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
  No custom NOC, no `DM_DYNAMIC_NOC`.
- **`opt_level`** — nothing in the directory sets it. DM kernels stay at the Metal 2.0 default `O2`
  (matching legacy). The **four** compute `KernelSpec`s (two core groups × two factories) each get an
  explicit `compiler_options.opt_level = KernelBuildOptLevel::O3`, because legacy `ComputeConfig`
  defaults to O3 while Metal 2.0's `CompilerOptions` defaults to O2.

---

# Deferred / Flagged

- **Brief item 9 inaccuracy** (new finding): the brief states GammaBetaGrad does not re-resolve the
  compute config inside the factory. It does, at `…gamma_beta_grad_program_factory.cpp:34-35`, exactly
  as InputGrad does. Both are translated from the re-resolved value. → report.
- **Brief item 1's "no entry for inputs" holds only because the op is BFLOAT16-only.** Confirmed from
  `check_tensor`'s default `data_types = {DataType::BFLOAT16}`; recorded here so a future dtype widening
  is known to need new `unpack_modes` entries. → report.
- The `_large` InputGrad path has **no committed test coverage**. Verified locally with an uncommitted
  large-shape test per invoker decision D2b; outcome recorded in the port report.
- Nothing else surfaced during planning. No structural issue the audit missed; no feature outside the
  audit's Appendix A; no capitulation trigger.
