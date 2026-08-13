# Port Plan — `moreh_group_norm_backward`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward`, ported from
`ProgramDescriptor` to Metal 2.0. Written during the inventory and planning steps; committed
alongside the port for review.

> **Read with `moreh_layer_norm_backward`'s plan.** This op owns **no** compute kernel — all three come
> from `moreh_layer_norm_backward`. The invoker assigned the bundled port (both ops, one branch/PR), so
> those three convert **in place, in their owner's directory** (shared-kernel Caution rung 3), and every
> atomic unit this op belongs to necessarily includes a layer-norm factory:
>
> | Unit | Factories converting together | Kernel sources |
> |---|---|---|
> | **A — gamma_beta_grad** | `MorehGroupNormBackwardGammaBetaGradFactory` + `MorehLayerNormBackwardGammaBetaGradFactory` | 5 |
> | **B — input_grad** | `MorehGroupNormBackwardInputGradFactory` + `MorehLayerNormBackwardInputGradFactory` | 8 |

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — both `DeviceOperation`s, one factory each.
- Variants: single factory per device-operation. `MorehGroupNormBackwardInputGradFactory` selects its
  **reader and compute kernel source at runtime** on `use_large_algorithm`, so that factory's atomic
  unit is 5 sources.
- Custom `compute_program_hash`: **none** — default reflection-based hash on both device-operations
  (`device/gamma_beta_grad/..._device_operation.cpp`, `device/input_grad/..._device_operation.cpp`; no
  `attribute_values` / `to_hash` backdoor either). Nothing to leave alone.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

---

### Variant: `MorehGroupNormBackwardGammaBetaGradFactory`

Source: `device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_factory.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/gamma_beta_grad/kernels/dataflow/reader_moreh_group_norm_backward_gamma_beta_grad.cpp` *(owned)* | `all_cores` | `gamma_grad_has_value`, then `TensorAccessorArgs` × 4 (output_grad, input, mean, rstd) | none | per core: `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf`, `tile_offset`, `num_channels_per_core`, `num_inner_tiles`, `num_channels`, `num_groups`, `origin_h`, `origin_w` | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/gamma_beta_grad/kernels/dataflow/writer_moreh_group_norm_backward_gamma_beta_grad.cpp` *(owned)* | `all_cores` | `gamma_grad_has_value`, `beta_grad_has_value`, then `TensorAccessorArgs` × 2 (gamma_grad, beta_grad — `nullptr` when absent) | none | per core: `gamma_grad_buf` **or literal `0u`**, `beta_grad_buf` **or `0u`**, `tile_offset`, `num_channels_per_core`, `num_inner_tiles`, `batch` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |
| compute_g1 | `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` **(borrowed)** | `core_group_1` | `num_channels_per_core_group_1`, `origin_h`, `origin_w`, `num_inner_tiles`, `Wt`, `gamma_grad_has_value`, `beta_grad_has_value`, `is_lastdim_layernorm` (=0), `is_groupnorm` (=1) | none | none | none | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_SCALAR` | unset → **O3** | `ComputeConfigDescriptor{}` — **all defaults** |
| compute_g2 | *(same source)* | `core_group_2` (only when non-empty) | `num_channels_per_core_group_2`, *rest identical* | none | none | none | *identical* | unset → **O3** | *identical* |

`grep -n opt_level` over the factory returns nothing → every kernel is at its legacy default.

#### CBs

`total_size = num_tiles * single_tile_size`, `core_ranges = all_cores`,
`page_size = single_tile_size`, `data_format = cb_data_format` (the output_grad dtype) on **every**
CB — this factory has no separate intermediate format. `tile` unset everywhere. A CB whose tile count
computes to 0 is **not** pushed (the `add_cb` lambda skips it).

| index | tiles | meaning |
|---|---|---|
| c_0 | 1 | output_grad (dy) |
| c_1 | 1 | input (x) |
| c_2 | 1 | mean |
| c_3 | 1 | rstd |
| c_4 | 1 | one — the reduce scaler; the kernel calls it `cb_scaler` |
| c_5 | `do_mask_h ? 1 : 0` | mask_h |
| c_6 | `do_mask_w ? 1 : 0` | mask_w — **this op is the only binder of c_6 on the shared compute kernel** |
| c_16 | `gamma_grad_has_value ? 1 : 0` | gamma_grad (dgamma) |
| c_17 | `beta_grad_has_value ? 1 : 0` | beta_grad (dbeta) |
| c_24 | 1 | y |
| c_25 | 1 | y·dy |
| c_26 | 1 | Add[dy] |
| c_27 | 1 | Add[y·dy] |
| c_28 | 1 | x − mean |
| c_29 | 1 | dycopy |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._gamma_beta_grad_factory.cpp:154` | `output_grad` | reader slot 0 |
| `..._gamma_beta_grad_factory.cpp:155` | `input` | reader slot 1 |
| `..._gamma_beta_grad_factory.cpp:156` | `mean` | reader slot 2 |
| `..._gamma_beta_grad_factory.cpp:157` | `rstd` | reader slot 3 |
| `..._gamma_beta_grad_factory.cpp:161` | `gamma_grad` (optional; `nullptr` buffer when absent) | writer slot 0 (literal `0u` when absent) |
| `..._gamma_beta_grad_factory.cpp:163` | `beta_grad` (optional; `nullptr` buffer when absent) | writer slot 1 (literal `0u` when absent) |

All **Case 1**; no raw base pointers, no 3rd (page-size) constructor argument.

#### Work split

- Driver: `tt_metal::split_work_to_cores(grid, num_channels)`
- `num_cores_to_be_used`, `all_cores`, `core_group_1` (`num_channels_per_core_group_1`),
  `core_group_2` (`num_channels_per_core_group_2`, may be empty).

---

### Variant: `MorehGroupNormBackwardInputGradFactory`

Source: `device/input_grad/moreh_group_norm_backward_input_grad_factory.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/input_grad/kernels/dataflow/reader_moreh_group_norm_backward_input_grad_{small,large}.cpp` *(owned; runtime-selected on `use_large_algorithm`)* | `all_cores` | `gamma_has_value`, then `TensorAccessorArgs` × 5 (output_grad, input, mean, rstd, gamma) | none | per core: `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf`, `gamma_buf` **or `0u`**, `tile_offset`, `num_rows_per_core`, `num_inner_tiles`, `num_channels`, `num_groups`, `origin_h`, `origin_w` | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/input_grad/kernels/dataflow/writer_moreh_group_norm_backward_input_grad.cpp` *(owned)* | `all_cores` | `TensorAccessorArgs` × 1 (input_grad) | none | per core: `input_grad_buf`, `tile_offset`, `num_rows_per_core`, `num_inner_tiles` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |
| compute_g1 | `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_{small,large}_kernel.cpp` **(borrowed; same runtime selection)** | `core_group_1` | `num_rows_per_core_group_1`, `origin_h`, `origin_w`, `num_inner_tiles`, `gamma_has_value`, `is_lastdim_layernorm` (=0), `is_groupnorm` (=1) | none | none | none | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_SCALAR` | unset → **O3** | `ComputeConfigDescriptor{}` — **all defaults** |
| compute_g2 | *(same source)* | `core_group_2` (only when non-empty) | `num_rows_per_core_group_2`, *rest identical* | none | none | none | *identical* | unset → **O3** | *identical* |

#### CBs

Same descriptor shape as gamma_beta_grad — `cb_data_format` on every CB, no separate intermediate
format. `use_large_algorithm` rewrites three tile counts before allocation (`im0_t = im1_t = 1`,
`im7_t = 0`).

| index | tiles (small) | tiles (large) | meaning |
|---|---|---|---|
| c_0 | 1 | 1 | output_grad (dy) |
| c_1 | 1 | 1 | input (x) |
| c_2 | 1 | 1 | mean |
| c_3 | 1 | 1 | rstd |
| c_4 | 1 | 1 | one / scaler |
| c_5 | 2 | 2 | inner_size(==n) — the kernel calls it `cb_n_recip_n` |
| c_6 | `gamma_has_value ? 1 : 0` | same | gamma |
| c_7 | `(do_mask_h \|\| do_mask_w) ? 2 : 0` | same | mask_h_w |
| c_16 | 1 | 1 | input_grad (dx) |
| c_24 | `num_inner_tiles` | 1 | dycopy |
| c_25 | `num_inner_tiles` | 1 | y |
| c_26 | 1 | 1 | Sum[dy] |
| c_27 | 1 | 1 | Sum[y·dy] |
| c_28 | 1 | 1 | small: recip_nrstd · large: tmp1 |
| c_29 | 1 | 1 | small: tmp1 · large: tmp2 |
| c_30 | 1 | 1 | small: tmp2 · large: tmp3 |
| c_31 | 1 | **0 (not allocated)** | small: tmp3 |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._input_grad_factory.cpp:174` | `output_grad` | reader slot 0 |
| `..._input_grad_factory.cpp:175` | `input` | reader slot 1 |
| `..._input_grad_factory.cpp:176` | `mean` | reader slot 2 |
| `..._input_grad_factory.cpp:177` | `rstd` | reader slot 3 |
| `..._input_grad_factory.cpp:178` | `gamma` (optional; `nullptr` buffer when absent) | reader slot 4 (literal `0u` when absent) |
| `..._input_grad_factory.cpp:181` | `input_grad` | writer slot 0 |

All **Case 1**; no 3rd constructor argument anywhere.

#### Work split

- Driver: `tt_metal::split_work_to_cores(grid, num_rows)` where `num_rows = n * num_groups`
- `num_cores_to_be_used`, `all_cores`, `core_group_1` (`num_rows_per_core_group_1`), `core_group_2`
  (`num_rows_per_core_group_2`, may be empty).

---

### Runtime kernel-source selection

`MorehGroupNormBackwardInputGradFactory` selects **both** its reader and its (borrowed) compute source
from one predicate, `use_large_algorithm`. Atomic unit: factory + `reader_..._small` +
`reader_..._large` + `writer_...` + the two borrowed compute sources = **5 sources**. The borrowed
compute sources' DFB → meaning map differs between small and large (c_28–c_31 above), so the DFB spec
names are derived per selected source path.

### Shared kernels

All three compute kernels are **borrowed** from `moreh_layer_norm_backward`:

| kernel | `_metal2` fork beside it? | consumers (`grep -rl <filename> ttnn/cpp/ttnn/operations/`) | rung |
|---|---|---|---|
| `.../moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | no | this op + its owner | **3 — convert in place** |
| `.../moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | no | this op + its owner | **3 — convert in place** |
| `.../moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | no | this op + its owner | **3 — convert in place** |

Rung 3 requires an explicit invoker assignment of the bundled port, not merely a consumer list. That
assignment is on record (invoker, 2026-08-13: both ops, one branch and PR — restated in the request that
opened this port), and the census confirms the assigned set is complete: exactly two consumers each.
Both binders convert in this change, so no fork is created and no pointer comment is needed.

**Two names conceded to the kernel's vocabulary** (the binding is named for the kernel, not for this
op's factory locals):

| this factory's local | kernel's name | binding |
|---|---|---|
| `one` (c_4, gamma_beta_grad and input_grad) | `cb_scaler` | **`scaler`** |
| `inner_size(==n)` (c_5, input_grad) | `cb_n_recip_n` | **`n_recip_n`** |

Likewise the compute CTAs at slots 3 / 4 take the kernel's names `NCHt` / `Wt`, not this factory's
`num_inner_tiles` / `Wt`.

**This op is the sole binder of `c_6` (mask_w) on the shared gamma_beta_grad compute kernel** — the
sibling op's `do_mask_w` is compile-time false (`is_groupnorm == false`), so it never allocates it. The
`#ifdef DO_MASK_W` branch is live on this side and dead on theirs; it must not be deleted.

### Flags

- No unreferenced kernel files: all 5 owned sources are bound by one of the two factories.
- No `GlobalCircularBuffer`, no aliased CB (`format_descriptors` is single-element on every descriptor),
  no semaphores, no `override_runtime_arguments`, no op-owned tensors, no varargs.
- Every dataflow kernel walks its runtime args with a running `get_arg_val<uint32_t>(i++)` counter, and
  the gamma_beta_grad / input_grad_small readers assign CB ids with a running `cb_id++` counter. Neither
  counter is a vararg signal — these are distinct fields read once each, in a fixed block, before any
  loop. Both counters disappear.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both factories). Neither
  ported-from factory has an `override_runtime_arguments`.
- **Custom `compute_program_hash`**: none — default reflection hash on both device-operations.
- **Implementation notes**: the port forces exactly one device-op-class edit per device-operation — the
  `create_descriptor` → `create_program_artifacts` signature change in
  `device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp:39` and
  `device/input_grad/moreh_group_norm_backward_input_grad_device_operation.hpp:32`. No pybind cleanup.
- **Unity-build hygiene**: `ttnn_op_moreh` is a unity-build target
  (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:7`) and **this op's two factories share one
  namespace** (`ttnn::operations::moreh::moreh_group_norm_backward`), so anonymous-namespace constants
  would collide outright. Spec-name constants are declared **function-local** inside
  `create_program_artifacts`.

---

## Planned Spec Shape

### Variant: `MorehGroupNormBackwardGammaBetaGradFactory`

- **KernelSpecs** (4): `reader`, `writer`, `compute_g1`, `compute_g2` (the last only when
  `core_group_2` is non-empty).
- **DataflowBufferSpecs**: `dy`, `x`, `mean`, `rstd`, `scaler`, `mask_h` *(iff `do_mask_h`)*, `mask_w`
  *(iff `do_mask_w`)*, `dgamma` *(iff `gamma_grad_has_value`)*, `dbeta` *(iff `beta_grad_has_value`)*,
  `y`, `ydy`, `dyadd`, `ydyadd`, `xmm`, `dycopy`. `entry_size = single_tile_size`, `num_entries` = the
  legacy tile count, `data_format_metadata = cb_data_format`, `tile_format_metadata` unset.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `output_grad`, `input`, `mean`, `rstd`, `gamma_grad` *(conditional)*,
  `beta_grad` *(conditional)*.
- **WorkUnitSpecs**: `wu_g1` = {reader, writer, compute_g1} on `core_group_1`; `wu_g2` = {reader,
  writer, compute_g2} on `core_group_2` (only when non-empty).
- **Op-owned tensors**: none.

### Variant: `MorehGroupNormBackwardInputGradFactory`

- **KernelSpecs** (4): `reader`, `writer`, `compute_g1`, `compute_g2`.
- **DataflowBufferSpecs**: `dy`, `x`, `mean`, `rstd`, `scaler`, `n_recip_n`, `gamma` *(iff
  `gamma_has_value`)*, `mask_h_w` *(iff `do_mask_h || do_mask_w`)*, `dx`, `dycopy`, `y`, `dysum`,
  `ydysum`, plus the path-dependent intermediates:
  - **small**: `recip_nrstd` (c_28), `tmp1` (c_29), `tmp2` (c_30), `tmp3` (c_31)
  - **large**: `tmp1` (c_28), `tmp2` (c_29), `tmp3` (c_30) — no fourth spec (`im7_t == 0`)
- **SemaphoreSpecs**: none.
- **TensorParameters**: `output_grad`, `input`, `mean`, `rstd`, `gamma` *(conditional)*, `input_grad`.
- **WorkUnitSpecs**: `wu_g1` / `wu_g2` as above.
- **Op-owned tensors**: none.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| gamma_beta_grad: `compute_desc_1` (core_group_1) + `compute_desc_2` (core_group_2), differing only on CTA `num_cols_per_core` | `compute_g1`, `compute_g2` of the borrowed `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | `wu_g1`, `wu_g2` | CONSUMER of `dy`/`x`/`mean`/`rstd`/`scaler`/`mask_h`/`mask_w`; PRODUCER of `dgamma`/`dbeta`; PRODUCER **and** CONSUMER (self-loop) of `y`/`ydy`/`dyadd`/`ydyadd`/`xmm`/`dycopy` |
| input_grad: `compute_desc_1` + `compute_desc_2`, differing only on CTA `num_rows_per_core` | `compute_g1`, `compute_g2` of the borrowed `..._input_grad_{small,large}_kernel.cpp` | `wu_g1`, `wu_g2` | CONSUMER of `dy`/`x`/`mean`/`rstd`/`scaler`/`n_recip_n`/`gamma`/`mask_h_w`; PRODUCER of `dx`; self-loop on every c_24–c_31 intermediate |

`core_group_1` and `core_group_2` are **disjoint**, so each node sees exactly one compute instance and
each shared DFB is an ordinary 1:1 there — the disjoint-node work split, **not** the same-grid
two-toucher case and **not** `allow_instance_multi_binding`. The per-group count stays a CTA.

---

## Dropped Plumbing

### gamma_beta_grad

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:154–157 | `TensorAccessorArgs(*<t>.buffer()).append_to(reader_cta)` × 4 | `TensorBinding` on the reader `KernelSpec` |
| factory:161–163 | `TensorAccessorArgs(<t> ? ... : nullptr).append_to(writer_cta)` × 2 | conditional `TensorBinding` on the writer `KernelSpec` |
| factory:245–248 | reader RTA slots 0–3 = the four `Buffer*` | `TensorBinding` (address auto-injected) |
| factory:259–268 | writer RTA slots 0–1 = `gamma_grad_buf` / `0u`, `beta_grad_buf` / `0u` | conditional `TensorBinding`; the literal-`0u` absent-optional sentinel disappears with it |
| factory:153 CTA 0 | `gamma_grad_has_value` (reader) | `compiler_options.defines["GAMMA_GRAD_HAS_VALUE"]` — it gates a conditional binding, so a define, not a named CTA |
| factory:159–160 CTA 0–1 | `gamma_grad_has_value`, `beta_grad_has_value` (writer) | `defines["GAMMA_GRAD_HAS_VALUE"]`, `defines["BETA_GRAD_HAS_VALUE"]` |
| factory:198–199 CTA 5–6 | `gamma_grad_has_value`, `beta_grad_has_value` (compute) | same two defines |
| *(new)* | the compute kernel derived `do_mask_h` / `do_mask_w` from CTAs; the legacy factory emitted no mask define at all | `defines["DO_MASK_H"]` / `["DO_MASK_W"]`, emitted to **every** kernel naming the resource (reader *and* compute) |
| reader kernel:33–40 | `uint32_t cb_id{0}; const auto cb_id_* = cb_id++;` × 7 | `dfb::dy` … `dfb::mask_w` |
| reader kernel:28–31, 82–91 | `TensorAccessorArgs<N>()` chain + 4 address RTAs | `TensorAccessor(tensor::<name>)` × 4 |
| writer kernel:28–30 | `uint32_t cb_id{16}; const auto cb_id_* = cb_id++;` × 2 | `dfb::dgamma`, `dfb::dbeta` |
| writer kernel:23–24, 34–38 | `TensorAccessorArgs<N>()` chain + 2 address RTAs | `TensorAccessor(tensor::<name>)` × 2 |
| both kernels | `int i{0}; get_arg_val<uint32_t>(i++)` run | `get_arg(args::<name>)`; the counter disappears |

### input_grad

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:174–178 | `TensorAccessorArgs(...)` × 5 (reader) | `TensorBinding` × 5 (`gamma` conditional) |
| factory:181 | `TensorAccessorArgs(*input_grad.buffer())` (writer) | `TensorBinding` |
| factory:265–273 | reader RTA slots 0–4 = four `Buffer*` + `gamma_buf` / `0u` | `TensorBinding`; the literal-`0u` sentinel disappears |
| factory:284 | writer RTA slot 0 = `input_grad_buf` | `TensorBinding` |
| factory:173 CTA 0 | `gamma_has_value` (reader) | `defines["GAMMA_HAS_VALUE"]` |
| factory:217 CTA 4 | `gamma_has_value` (compute) | `defines["GAMMA_HAS_VALUE"]` |
| *(new)* | the borrowed compute kernels derive `do_mask_h` / `do_mask_w` from CTAs; the reader derives them from its `origin_h` / `origin_w` RTAs | `defines["DO_MASK_H"]` / `["DO_MASK_W"]` to reader **and** compute |
| reader kernels:35–43 | `uint32_t cb_id{0}; const auto cb_id_* = cb_id++;` × 8 | `dfb::dy` … `dfb::mask_h_w` |
| reader kernels:29–33, 82–95 | `TensorAccessorArgs<N>()` chain + 5 address RTAs | `TensorAccessor(tensor::<name>)` |
| writer kernel:18–29 | `TensorAccessorArgs<0>()` + address RTA; `uint32_t cb_id{16}; cb_id++` | `TensorAccessor(tensor::input_grad)`; `dfb::dx` |
| all kernels | `int i{0}; get_arg_val<uint32_t>(i++)` run | `get_arg(args::<name>)` |

**Retained CTAs** (ordinary scalars the kernel computes with, not binding selectors, so they stay as
*named* CTAs): `num_cols_per_core` / `num_rows_per_core`, `origin_H`, `origin_W`, `NCHt`, `Wt`,
`is_lastdim_layernorm`, `is_groupnorm`. **Retained RTAs**: every remaining scalar, all named, none
varargs.

---

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the compute-only intermediates (gamma_beta_grad c_24–c_29; input_grad c_24–c_31, one fewer under the
  large algorithm) have exactly **one** toucher, so each compute `KernelSpec` binds them as both
  PRODUCER and CONSUMER under a single `accessor_name`. Census re-derived from the kernel bodies; it
  agrees with the brief. The readers' `get_write_ptr()` calls sit between the *same kernel's*
  `reserve_back` and `push_back` — a public peek on a binding it already holds, not a second endpoint.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings):
  nine resources across the two factories (`mask_h`, `mask_w`, `dgamma`, `dbeta`, `gamma`, `mask_h_w`,
  and tensors `gamma_grad`, `beta_grad`, `gamma`). Host binds conditionally, emits the matching define
  to **every** kernel naming the resource, kernel `#ifdef`-gates construction and every reference.
  This op reaches the absent-`dgamma` configuration most directly — it already allocates c_16/c_17
  conditionally — so both binders must emit consistent defines for the shared kernel.
- [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names):
  in the borrowed input_grad compute kernels — one `DataflowBufferSpec`, one `DFBBinding` and one
  `DataflowBuffer` object per FIFO, working names as handle aliases. Not `advanced_options.alias_with`.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  including `generate_mask_h_w(dfb_mask_h_w, mask_h, mask_w, dfb_mask_h_w.get_tile_size())` — the
  free `get_tile_size(cb_id)` moves onto the object before being handed to the donor helper.
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  rung **3** (convert in place, in the owner's directory) on all three borrowed compute kernels, under
  the invoker's explicit bundled assignment.
- [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  — avoided.
- [Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  — the `get_arg_val<uint32_t>(i++)` runs in all five owned dataflow kernels are the *trap* shape, not
  a vararg signal. Every one becomes a named arg.
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  function-local spec-name constants — mandatory here, since this op's two factories share a namespace
  inside a unity-build target.

---

## Deferred / Flagged

- **New findings during planning**:
  - **Style B compute config, confirmed against the resolved values.** Both factories set
    `ComputeConfigDescriptor{}` with no fields — so the Metal 2.0 side builds a `ComputeGen1Config`
    **directly** and lets its defaults stand. Routing through `ttnn::to_compute_hardware_config` would
    silently flip every field, because the TTNN helper's defaults are the high-performance ones and the
    Metal struct's are the high-precision ones. With `enable_32_bit_dest` left at its default `false`,
    the newly-required Float32 `unpack_modes` entry rule does not fire, so `unpack_modes` stays empty —
    confirmed against the resolved config, not assumed. This is the one place the op differs materially
    from `moreh_layer_norm_backward`, which is Style A.
  - Three pre-existing oddities preserved as-is (see the audit's Misc anomalies, and the port report):
    the all-default `ComputeConfigDescriptor{}` despite the op carrying a `compute_kernel_config`
    attribute; `num_groups` / `num_channels` / `origin_h` / `origin_w` / `num_inner_tiles` riding as
    per-core RTAs despite being cache-key-invariant; and the literal-`0u` absent-optional sentinel where
    the sibling op passes a null `Buffer*`.
