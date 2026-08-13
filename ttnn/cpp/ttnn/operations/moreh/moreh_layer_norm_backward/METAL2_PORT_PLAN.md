# Port Plan — `moreh_layer_norm_backward`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward`, ported from
`ProgramDescriptor` to Metal 2.0. Written during the inventory and planning steps; committed
alongside the port for review.

> **Read with `moreh_group_norm_backward`'s plan.** The atomic units span both ops — group-norm
> binds three of this op's compute kernels, and the invoker assigned the bundled port (both ops,
> one branch/PR), so those three kernels convert **in place** (shared-kernel Caution rung 3).
>
> | Unit | Factories converting together | Kernel sources |
> |---|---|---|
> | **A — gamma_beta_grad** | `MorehLayerNormBackwardGammaBetaGradFactory` + `MorehGroupNormBackwardGammaBetaGradFactory` | 5 |
> | **B — input_grad** | `MorehLayerNormBackwardInputGradFactory` + `MorehGroupNormBackwardInputGradFactory` | 8 |

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — both `DeviceOperation`s, one factory each.
- Variants: single factory per device-operation. `MorehLayerNormBackwardInputGradFactory` selects its
  **reader and compute kernel source at runtime** on `use_large_algorithm` (an L1-budget decision), so
  that factory's atomic unit is 5 sources, not 3.
- Custom `compute_program_hash`: **none** — default reflection-based hash on both device-operations
  (`device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.cpp`,
  `device/moreh_layer_norm_backward_input_grad_device_operation.cpp`; no `attribute_values` / `to_hash`
  backdoor either). Nothing to leave alone.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

---

### Variant: `MorehLayerNormBackwardGammaBetaGradFactory`

Source: `device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_layer_norm_backward_gamma_beta_grad.cpp` | `all_cores` | `gamma_grad_has_value`, `do_mask_h`, then `TensorAccessorArgs` × 4 (output_grad, input, mean, rstd) | none | per core: `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf`, `num_cols_per_core`, `num_outer`, `num_inner`, `tile_offset`, `mask_h`, `normalized_dims`, `mean_rstd_height`, `mean_rstd_width` | none | `FP32_DEST_ACC_EN` (iff `fp32_dest_acc_en`) | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_moreh_layer_norm_backward_gamma_beta_grad.cpp` | `all_cores` | `gamma_grad_has_value`, `beta_grad_has_value`, then `TensorAccessorArgs` × 2 (gamma_grad, beta_grad — `nullptr` when absent) | none | per core: `gamma_grad_buf`, `beta_grad_buf`, `num_cols_per_core`, `tile_offset` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | `core_group_1` | `num_cols_per_core_group_1`, `origin_H`, `origin_W`, `num_outer`, `num_inner`, `gamma_grad_has_value`, `beta_grad_has_value`, `is_lastdim_layer_norm`, `is_groupnorm` | none | none | none | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_COL`, `FP32_DEST_ACC_EN` (iff) | unset → **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute_g2 | *(same source)* | `core_group_2` (only when non-empty) | `num_cols_per_core_group_2`, *rest identical* | none | none | none | *identical* | unset → **O3** | *identical* |

`grep -n opt_level` over the factory returns nothing → every kernel is at its legacy default.

#### CBs

`total_size = num_tiles * tile_size(fmt)`, `core_ranges = all_cores`, `page_size = tile_size(fmt)`,
`tile` unset on every descriptor. `fmt` is `cb_data_format` (the output_grad dtype) for c_0–c_17 and
`intermed_cb_format` (`Float32` iff `fp32_dest_acc_en`, else `cb_data_format`) for c_24–c_29. A CB
whose tile count computes to 0 is **not** pushed (the `push_cb` lambda skips it).

| index | tiles | fmt | meaning |
|---|---|---|---|
| c_0 | 1 | data | output_grad (dy) |
| c_1 | 1 | data | input (x) |
| c_2 | 1 | data | mean |
| c_3 | 1 | data | rstd |
| c_4 | 1 | data | scaler |
| c_5 | `do_mask_h ? 1 : 0` | data | mask_h |
| c_16 | 1 | data | gamma_grad (dgamma) |
| c_17 | 1 | data | beta_grad (dbeta) |
| c_24 | 1 | intermed | y |
| c_25 | 1 | intermed | y·dy |
| c_26 | 1 | intermed | Add[dy] |
| c_27 | 1 | intermed | Add[y·dy] |
| c_28 | 1 | intermed | x − mean |
| c_29 | 1 | intermed | dycopy |

**c_6 (mask_w) is never allocated by this factory** — the shared compute kernel constructs a
`DataflowBuffer` on it unconditionally but only uses it under `do_mask_w`, which is compile-time false
here (`is_groupnorm == false`). `moreh_group_norm_backward` *does* allocate it, so the branch is live.

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._gamma_beta_grad_program_factory.cpp:145` | `output_grad` | reader slot 0 |
| `..._gamma_beta_grad_program_factory.cpp:146` | `input` | reader slot 1 |
| `..._gamma_beta_grad_program_factory.cpp:147` | `mean` | reader slot 2 |
| `..._gamma_beta_grad_program_factory.cpp:148` | `rstd` | reader slot 3 |
| `..._gamma_beta_grad_program_factory.cpp:152` | `gamma_grad` (optional; `nullptr` buffer when absent) | writer slot 0 |
| `..._gamma_beta_grad_program_factory.cpp:153` | `beta_grad` (optional; `nullptr` buffer when absent) | writer slot 1 |

All **Case 1** (accessor-mediated); no raw base pointers, no 3rd (page-size) constructor argument.

#### Work split

- Driver: `tt::tt_metal::split_work_to_cores(grid, num_inner)`
- `num_cores`, `all_cores`, `core_group_1` (`num_cols_per_core_group_1`), `core_group_2`
  (`num_cols_per_core_group_2`, may be empty).

---

### Variant: `MorehLayerNormBackwardInputGradFactory`

Source: `device/moreh_layer_norm_backward_input_grad_program_factory.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_layer_norm_backward_input_grad_{small,large}.cpp` *(runtime-selected on `use_large_algorithm`)* | `all_cores` | `gamma_has_value`, `do_mask_h`, `do_mask_w`, then `TensorAccessorArgs` × 5 (output_grad, input, mean, rstd, gamma) | none | per core: `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf`, `gamma_buf`, `num_rows_per_core`, `num_inner`, `tile_offset`, `n`, `recip_n`, `mask_h`, `mask_w`, `normalized_dims`, `mean_rstd_height`, `mean_rstd_width` | none | `FP32_DEST_ACC_EN` (iff) | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_moreh_layer_norm_backward_input_grad.cpp` | `all_cores` | `TensorAccessorArgs` × 1 (input_grad) | none | per core: `input_grad_buf`, `num_rows_per_core`, `num_inner`, `tile_offset` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/moreh_layer_norm_backward_input_grad_{small,large}_kernel.cpp` *(same runtime selection)* | `core_group_1` | `num_rows_per_core_group_1`, `origin_H`, `origin_W`, `num_inner`, `gamma_has_value`, `is_lastdim_layer_norm`, `is_groupnorm` | none | none | none | `REDUCE_OP=PoolType::AVG`, `REDUCE_DIM=REDUCE_ROW` (lastdim) or `REDUCE_SCALAR`, `FP32_DEST_ACC_EN` (iff) | unset → **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute_g2 | *(same source)* | `core_group_2` (only when non-empty) | `num_rows_per_core_group_2`, *rest identical* | none | none | none | *identical* | unset → **O3** | *identical* |

#### CBs

Same descriptor shape as gamma_beta_grad. `use_large_algorithm` rewrites three tile counts before
allocation (`im0_t = im1_t = 1`, `im7_t = 0`).

| index | tiles (small) | tiles (large) | fmt | meaning |
|---|---|---|---|---|
| c_0 | 1 | 1 | data | output_grad (dy) |
| c_1 | 1 | 1 | data | input (x) |
| c_2 | 1 | 1 | data | mean |
| c_3 | 1 | 1 | data | rstd |
| c_4 | 1 | 1 | data | scaler |
| c_5 | 2 | 2 | data | n_recip_n |
| c_6 | `gamma_has_value ? 1 : 0` | same | data | gamma |
| c_7 | `(do_mask_h \|\| do_mask_w) ? 2 : 0` | same | data | mask_h_w |
| c_16 | 1 | 1 | data | input_grad (dx) |
| c_24 | `num_inner` | 1 | intermed | dycopy |
| c_25 | `num_inner` | 1 | intermed | y |
| c_26 | 1 | 1 | intermed | Sum[dy] |
| c_27 | 1 | 1 | intermed | Sum[y·dy] |
| c_28 | 1 | 1 | intermed | small: recip_nrstd · large: tmp1 |
| c_29 | 1 | 1 | intermed | small: tmp1 · large: tmp2 |
| c_30 | 1 | 1 | intermed | small: tmp2 · large: tmp3 |
| c_31 | 1 | **0 (not allocated)** | intermed | small: tmp3 |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._input_grad_program_factory.cpp:180` | `output_grad` | reader slot 0 |
| `..._input_grad_program_factory.cpp:181` | `input` | reader slot 1 |
| `..._input_grad_program_factory.cpp:182` | `mean` | reader slot 2 |
| `..._input_grad_program_factory.cpp:183` | `rstd` | reader slot 3 |
| `..._input_grad_program_factory.cpp:184` | `gamma` (optional; `nullptr` buffer when absent) | reader slot 4 |
| `..._input_grad_program_factory.cpp:187` | `input_grad` | writer slot 0 |

All **Case 1**; no 3rd constructor argument anywhere.

#### Work split

- Driver: `tt::tt_metal::split_work_to_cores(grid, num_outer)`
- `num_cores`, `all_cores`, `core_group_1` (`num_rows_per_core_group_1`), `core_group_2`
  (`num_rows_per_core_group_2`, may be empty).

---

### Runtime kernel-source selection

`MorehLayerNormBackwardInputGradFactory` selects **both** its reader and its compute source from one
predicate, `use_large_algorithm` (`cb_usage >= available_L1`). The two axes move together, so the
factory's atomic unit is: factory + `reader_..._small` + `reader_..._large` + `writer_...` +
`..._input_grad_small_kernel` + `..._input_grad_large_kernel` = **5 sources**.

The DFB → meaning map **differs between the two compute sources** (see the c_28–c_31 rows above), so
the DFB spec names are derived per selected source path, not once for the factory.

### Shared kernels

Three kernel sources in this op's own directory are **lent** to `moreh_group_norm_backward`:

| kernel | `_metal2` fork beside it? | consumers (`grep -rl <filename> ttnn/cpp/ttnn/operations/`) | rung |
|---|---|---|---|
| `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | no | this op + `moreh_group_norm_backward` (gamma_beta_grad factory) | **3 — convert in place** |
| `device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp` | no | this op + `moreh_group_norm_backward` (input_grad factory) | **3 — convert in place** |
| `device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp` | no | this op + `moreh_group_norm_backward` (input_grad factory) | **3 — convert in place** |

Rung 3 requires an explicit invoker assignment of the bundled port, not merely a consumer list. That
assignment is on record (invoker, 2026-08-13: both ops, one branch and PR — restated in the request that
opened this port), and the census confirms the assigned set is the *complete* set: exactly two consumers
each, no third binder anywhere in the tree. Both binders convert in this change, so no fork is created
and no pointer comment is needed.

**Binding vocabulary is named for the kernel, not for either factory's locals.** The one name the two
ops disagree on is `c_4`: layer-norm's factory calls it *scaler*, group-norm's calls it *one*. The
kernel says `cb_scaler` → the binding is **`scaler`**. Same rule gives `NCHt` / `Wt` for the compute
CTAs at slots 3 / 4, which the two factories fill with differently-named locals.

### Flags

- No unreferenced kernel files: all 8 sources in `device/kernels/` are bound by one of the two factories.
- The gamma_beta_grad reader and both input_grad readers carry a **file-local** `read_mean_rstd`
  template taking a `uint32_t cb_id`. It is inside this op's directory (an ordinary in-scope edit), and
  its `get_tile_size(cb_id)` moves onto the constructed object.
- No `GlobalCircularBuffer`, no aliased CB (`format_descriptors` is single-element on every descriptor),
  no semaphores, no `override_runtime_arguments`, no op-owned tensors, no varargs.
- Both input_grad compute kernels reach one FIFO through several working names (`cb_xmm`, `cb_dyadd`,
  `cb_ydy`, …) — **same-FIFO aliasing**, not aliased DFBs. See Applied Patterns.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both factories). Neither
  ported-from factory has an `override_runtime_arguments`.
- **Custom `compute_program_hash`**: none — default reflection hash on both device-operations.
- **Implementation notes**: the port forces exactly one device-op-class edit per device-operation — the
  `create_descriptor` → `create_program_artifacts` signature change in
  `device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp:34` and
  `device/moreh_layer_norm_backward_input_grad_device_operation.hpp:33`. No pybind cleanup: the nanobind
  file exposes only the user-facing op, never `create_descriptor`.
- **Unity-build hygiene**: `ttnn_op_moreh` is a unity-build target
  (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:7`), and `moreh_group_norm_backward`'s two factories
  share one namespace. Spec-name constants are declared **function-local** inside
  `create_program_artifacts` rather than in an anonymous namespace, so no two factories can collide.

---

## Planned Spec Shape

### Variant: `MorehLayerNormBackwardGammaBetaGradFactory`

- **KernelSpecs** (4, mirroring the legacy 4): `reader`, `writer`, `compute_g1`, `compute_g2`
  (the last only when `core_group_2` is non-empty).
- **DataflowBufferSpecs** (one per allocated legacy CB): `dy`, `x`, `mean`, `rstd`, `scaler`,
  `mask_h` *(iff `do_mask_h`)*, `dgamma` *(iff `gamma_grad_has_value`)*, `dbeta` *(iff
  `beta_grad_has_value`)*, `y`, `ydy`, `dyadd`, `ydyadd`, `xmm`, `dycopy`.
  `entry_size = tile_size(fmt)`, `num_entries` = the legacy tile count, `data_format_metadata = fmt`;
  `tile_format_metadata` left unset (the legacy `CBFormatDescriptor::tile` was never set).
  `mask_w` is **not** declared — this factory never allocated c_6.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `output_grad`, `input`, `mean`, `rstd`, `gamma_grad` *(conditional)*,
  `beta_grad` *(conditional)*.
- **WorkUnitSpecs**: `wu_g1` = {reader, writer, compute_g1} on `core_group_1`; `wu_g2` = {reader,
  writer, compute_g2} on `core_group_2` (only when non-empty).
- **Op-owned tensors**: none.

### Variant: `MorehLayerNormBackwardInputGradFactory`

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
| gamma_beta_grad: `compute_desc_1` (core_group_1) + `compute_desc_2` (core_group_2), differing only on CTA `num_cols_per_core` | `compute_g1`, `compute_g2` of `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` | `wu_g1`, `wu_g2` | CONSUMER of `dy`/`x`/`mean`/`rstd`/`scaler`/`mask_h`; PRODUCER of `dgamma`/`dbeta`; PRODUCER **and** CONSUMER (self-loop) of `y`/`ydy`/`dyadd`/`ydyadd`/`xmm`/`dycopy` |
| input_grad: `compute_desc_1` + `compute_desc_2`, differing only on CTA `num_rows_per_core` | `compute_g1`, `compute_g2` of the selected `..._input_grad_{small,large}_kernel.cpp` | `wu_g1`, `wu_g2` | CONSUMER of `dy`/`x`/`mean`/`rstd`/`scaler`/`n_recip_n`/`gamma`/`mask_h_w`; PRODUCER of `dx`; self-loop on every c_24–c_31 intermediate |

The two node sets are **disjoint** (`core_group_1` ∩ `core_group_2` = ∅), so each node sees exactly one
compute instance and each shared DFB is an ordinary 1:1 there — this is the disjoint-node work split,
**not** the same-grid two-toucher case, and **not** `allow_instance_multi_binding`. The per-group count
stays a CTA on each `KernelSpec`; demoting it to an RTA is the anti-pattern this row exists to prevent.

---

## Dropped Plumbing

### gamma_beta_grad

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:145–148 | `TensorAccessorArgs(<t>.buffer()).append_to(reader_cta)` × 4 | `TensorBinding` on the reader `KernelSpec` |
| factory:152–153 | `TensorAccessorArgs(<t>->buffer() : nullptr).append_to(writer_cta)` × 2 | conditional `TensorBinding` on the writer `KernelSpec` |
| factory:268–271 | reader RTA slots 0–3 = `output_grad_buf`, `input_buf`, `mean_buf`, `rstd_buf` | `TensorBinding` (address auto-injected) |
| factory:281 | writer RTA slots 0–1 = `gamma_grad_buf`, `beta_grad_buf` (`nullptr` when absent) | conditional `TensorBinding` |
| factory:144 CTA 0 | `gamma_grad_has_value` (reader) | `compiler_options.defines["GAMMA_GRAD_HAS_VALUE"]` — it gates a conditional binding, so it becomes a define, not a named CTA |
| factory:144 CTA 1 | `do_mask_h` (reader) | `compiler_options.defines["DO_MASK_H"]` |
| factory:151 CTA 0–1 | `gamma_grad_has_value`, `beta_grad_has_value` (writer) | `defines["GAMMA_GRAD_HAS_VALUE"]`, `defines["BETA_GRAD_HAS_VALUE"]` |
| factory:203–204 CTA 5–6 | `gamma_grad_has_value`, `beta_grad_has_value` (compute) | same two defines |
| reader kernel:115–120 | `constexpr uint32_t cb_id_* = 0..5;` | `dfb::dy` … `dfb::mask_h` |
| reader kernel:124–132 | `TensorAccessorArgs<N>()` chain + 4 address RTAs | `TensorAccessor(tensor::<name>)` × 4 |
| writer kernel:19–26 | `TensorAccessorArgs<N>()` chain + 2 address RTAs | `TensorAccessor(tensor::<name>)` × 2 |
| writer kernel:22–23 | `constexpr uint32_t cb_id_gamma_grad = 16; … = 17;` | `dfb::dgamma`, `dfb::dbeta` |
| compute kernel:20–54 | `constexpr auto cb_* = tt::CBIndex::c_N;` × 15 | `dfb::<name>` |
| compute kernel:64, 68 | `constexpr bool do_mask_h/do_mask_w = …` (CTA-derived) | `DO_MASK_H` / `DO_MASK_W` defines — the gate must reach the *preprocessor*, and it must be fed to **every** kernel that names the resource |
| all three kernels | positional `get_compile_time_arg_val(N)` / `get_arg_val<uint32_t>(N)` | `get_arg(args::<name>)` |

### input_grad

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:180–184 | `TensorAccessorArgs(...)` × 5 (reader) | `TensorBinding` × 5 (`gamma` conditional) |
| factory:187 | `TensorAccessorArgs(input_grad.buffer())` (writer) | `TensorBinding` |
| factory:310–314 | reader RTA slots 0–4 = the five `Buffer*` | `TensorBinding` |
| factory:326 | writer RTA slot 0 = `input_grad_buf` | `TensorBinding` |
| factory:178–179 CTA 0–2 | `gamma_has_value`, `do_mask_h`, `do_mask_w` (reader) | `defines["GAMMA_HAS_VALUE"]`, `["DO_MASK_H"]`, `["DO_MASK_W"]` |
| factory:245 CTA 4 | `gamma_has_value` (compute) | `defines["GAMMA_HAS_VALUE"]` |
| reader kernels:116–123 | `constexpr uint32_t cb_id_* = 0..7;` | `dfb::dy` … `dfb::mask_h_w` |
| reader kernels:128–140 | `TensorAccessorArgs<N>()` chain + 5 address RTAs | `TensorAccessor(tensor::<name>)` |
| writer kernel:16–20 | `TensorAccessorArgs<0>()` + address RTA; `constexpr uint32_t cb_id_input_grad = 16;` | `TensorAccessor(tensor::input_grad)`; `dfb::dx` |
| compute kernels:24–60 (small) / 24–57 (large) | `constexpr auto cb_* = tt::CBIndex::c_N;` | `dfb::<name>` |
| compute kernels:70, 73 | `constexpr bool do_mask_h/do_mask_w` (CTA-derived) | `DO_MASK_H` / `DO_MASK_W` defines |
| all four kernels | positional CTAs / RTAs | `get_arg(args::<name>)` |

**Retained CTAs** (they are ordinary scalars the kernel computes with, not binding selectors, so they
stay as *named* CTAs): `num_cols_per_core` / `num_rows_per_core`, `origin_H`, `origin_W`, `NCHt`, `Wt`,
`is_lastdim_layernorm`, `is_groupnorm`. **Retained RTAs**: every remaining scalar, all named, none
varargs — each is a distinct field read once in a fixed block at the top of `kernel_main`.

---

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the compute-only intermediates — gamma_beta_grad `y`/`ydy`/`dyadd`/`ydyadd`/`xmm`/`dycopy`, input_grad
  `dycopy`/`y`/`dysum`/`ydysum`/`recip_nrstd`/`tmp1`/`tmp2`/`tmp3` — have exactly **one** toucher (the
  compute kernel), so each compute `KernelSpec` binds them as both PRODUCER and CONSUMER under a single
  `accessor_name`. Census re-derived from the kernel bodies; it agrees with the brief.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings):
  nine resources across the two factories (`mask_h`, `mask_w`, `dgamma`, `dbeta`, `gamma`, `mask_h_w`,
  and tensors `gamma_grad`, `beta_grad`, `gamma`). Host binds conditionally, emits the matching
  `compiler_options.defines` flag to **every** kernel naming the resource, and the kernel
  `#ifdef`-gates the `DataflowBuffer` / `TensorAccessor` construction and every expression referencing
  it. The `cb_out_init` ternary at
  `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:73` resolves **both** operands at
  parse time, so the ternary itself is gated, not just its uses.
- [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names):
  both input_grad compute kernels reach `tmp1`/`tmp2`/`tmp3` (and, in the large kernel, `y`) through
  working names — `cb_xmm`, `cb_dyadd`, `cb_ydy`, `cb_ydyadd`, `cb_ndy`, `cb_ndymdysum`, `cb_yydysum`,
  `cb_tmp4`, `cb_recip_nrstd`. **One** `DataflowBufferSpec`, **one** `DFBBinding` and **one**
  `DataflowBuffer` object per FIFO; each working name becomes a `constexpr` handle alias plus a
  reference alias to the single object. Not `advanced_options.alias_with` — that models distinct
  buffers sharing memory and would break the shared-pointer coherence these names rely on.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `copy_tile` / `add_tiles` / `mul_tiles_bcast_*` / `compute_kernel_hw_startup` take `dfb::<name>`
  directly; `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, in, scaler, out>` takes them in
  **non-type template parameter** position, which the `constexpr operator uint32_t` supports.
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  rung **3** (convert in place) on all three lent compute kernels, under the invoker's explicit bundled
  assignment. See Shared kernels above.
- [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  — avoided: two compute `KernelSpec`s in two `WorkUnitSpec`s, per-group count kept as a CTA.
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  spec-name constants are function-local, not anonymous-namespace, because `ttnn_op_moreh` is a
  unity-build target shared with every other moreh op.

---

## Deferred / Flagged

- **New findings during planning**:
  - The legacy gamma_beta_grad factory allocates `c_16`/`c_17` **unconditionally** (`out0_t = out1_t = 1`)
    even when the matching optional output is absent, so in that configuration the CB is allocated with
    zero touchers. Under Metal 2.0 the binding is conditional, so the DFB is simply not declared —
    a config-scoped dead-CB drop, zero functional change (a bindingless DFB would be rejected by the
    validator regardless). `moreh_group_norm_backward`'s sibling factory already allocates 0 tiles there.
  - `unpack_modes` is **newly required** where legacy had nothing: under `fp32_dest_acc_en` the
    intermediates are `Float32` and `enable_32_bit_dest` is true, so every Float32 DFB the compute kernel
    *consumes* needs an explicit entry. The legacy config set no `unpack_to_dest_mode`, so every entry is
    the legacy default → `UnpackMode::UnpackToSrc`. Derived, not guessed.
  - Two pre-existing oddities preserved as-is (see the audit's Misc anomalies, and the port report):
    input_grad compute CTA slot 3 is named `Wt` but carries `num_inner`; and `normalized_dims` /
    `mean_rstd_height` / `mean_rstd_width` ride as per-core RTAs despite being cache-key-invariant.
  - `moreh_layer_norm_backward_input_grad_small_kernel.cpp:478-480` ends with a second
    `wait_front(2)` on `mask_h_w` where the large kernel has `pop_front(2)`. Preserved verbatim;
    written up in the port report.
