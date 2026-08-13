# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port` *(carry this line into the port report's Provenance section)*

**Two factories, two DeviceOperations, one directory.** They share no kernels with each other. Port them
as two units:

| Factory | Kernel sources it binds | Atomic unit size |
|---|---|---|
| `MorehGroupNormBackwardGammaBetaGradFactory` | reader *(owned)*, writer *(owned)*, compute **(borrowed)** | 3 |
| `MorehGroupNormBackwardInputGradFactory` | reader **small\|large** *(owned)*, writer *(owned)*, compute **small\|large** **(borrowed)** | **5** — runtime-selected on `use_large_algorithm`; all five convert together |

> **This op owns no compute kernels.** All three come from `moreh_layer_norm_backward`. Read that op's
> brief alongside this one — it carries the compute kernels' internal analysis (self-loop intermediates,
> same-FIFO aliasing, the `cb_out_init` ternary). See [Watch for → Cross-op / shared kernels](#watch-for).

## ⚠ Your atomic unit spans both ops — this op cannot be ported alone

**The bundled port is confirmed** (invoker, 2026-08-13): the three borrowed compute kernels convert **in
place** in `moreh_layer_norm_backward`'s directory, no forks. That has a consequence which overrides the
recipe's usual "port one factory at a time" default.

A shared kernel converted in place flips to Metal 2.0 **once**, and at that moment every factory binding
it must already speak Metal 2.0 bindings or the build breaks. Since this op owns **no** compute kernel,
every unit it belongs to necessarily includes a layer-norm factory:

| Unit | Factories that must convert together | Kernel sources |
|---|---|---|
| **A — gamma_beta_grad** | `MorehGroupNormBackwardGammaBetaGradFactory` **+** `MorehLayerNormBackwardGammaBetaGradFactory` | 5 |
| **B — input_grad** | `MorehGroupNormBackwardInputGradFactory` **+** `MorehLayerNormBackwardInputGradFactory` | 8 |

A and B share no kernel, so they are independent: either order, separate sessions, each separately
buildable and testable. **Within** a unit there is no smaller buildable increment — a long stretch with
no green build is the expected shape here, not a stop signal. Unit B is the bigger one (two algorithm
paths, and the same-FIFO alias map differs between small and large). If a unit overruns your budget,
hand the **whole** unit to a fresh primary instance to continue from `METAL2_PORT_PLAN.md`; never leave
a half-converted unit.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept` (both)
- **Custom `compute_program_hash`:** none — default reflection hash. Nothing to leave alone.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none`
  `TensorParameter relaxation` · `get_dynamic_runtime_args`. Also absent, though none of them gate: a
  custom hash, an `override_runtime_arguments`, a pybound `create_descriptor`.
- **Device-op-class edit forced:** only the `create_descriptor` → `create_program_artifacts` signature
  change in `device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp:39`
  and `device/input_grad/moreh_group_norm_backward_input_grad_device_operation.hpp:32`. No pybind
  cleanup.

## Construct — to do

**Tensor bindings** (per binding) — **all Case 1**:

*gamma_beta_grad*
- `output_grad`, `input`, `mean`, `rstd` — Case 1 → `TensorParameter` / `TensorBinding`; reader uses
  `TensorAccessor(tensor::…)`.
- `gamma_grad` — Case 1, **conditional** on `gamma_grad_has_value`.
- `beta_grad` — Case 1, **conditional** on `beta_grad_has_value`.

*input_grad*
- `output_grad`, `input`, `mean`, `rstd`, `input_grad` — Case 1.
- `gamma` — Case 1, **conditional** on `gamma_has_value`.

The `TensorAccessorArgs(...).append_to(...)` chains on the host, the `TensorAccessorArgs<N>()` /
`next_compile_time_args_offset()` chains in the kernels, and the `Buffer*` / literal-`0u` RTA slots all
disappear.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none — all 11 accessor sites are the 2-arg form.

**CB endpoints:**

- **Self-loop** (compute-only intermediate, one toucher — bind the compute `KernelSpec` as both
  PRODUCER and CONSUMER):
  - gamma_beta_grad: `c_24` y, `c_25` y·dy, `c_26` Add[dy], `c_27` Add[y·dy], `c_28` x−mean, `c_29` dycopy
  - input_grad **small**: `c_24`–`c_31` (dycopy, y, Sum[dy], Sum[y·dy], recip_nrstd, tmp1, tmp2, tmp3)
  - input_grad **large**: `c_24`–`c_30` *(no `c_31` — `im7_t = 0` under the large algorithm; correct,
    not a dead CB)*
- **Legal 1:1** (reader→compute, or compute→writer): everything else.
- **Multi-binding flag:** **none anywhere.** Every CB has at most two touchers, each locked to one FIFO
  role. The `get_write_ptr()` calls in your readers sit between that kernel's own `reserve_back` and
  `push_back` — a public peek on a binding it already holds, **not** a second endpoint. If you find
  yourself reaching for `allow_instance_multi_binding`, recount.
- **Dead-CB drop:** none. This op already allocates every conditional CB with 0 tiles when unused, so
  there is no allocated-but-untouched buffer in any configuration.

**Conditional bindings — the bulk of the port.** Host binds conditionally + emits a
`compiler_options.defines` flag + kernel `#ifdef`-gates the alias and every expression naming it. All of
these host booleans already exist in the factory, so the promotion is a direct lift:

| Resource | Condition | Kernels to gate |
|---|---|---|
| `dfb::mask_h` (`c_5`) | `do_mask_h` | reader gbg, compute gbg *(borrowed)* |
| `dfb::mask_w` (`c_6`) | `do_mask_w` | reader gbg, compute gbg *(borrowed)* |
| `dfb::dgamma` (`c_16`) | `gamma_grad_has_value` | compute gbg *(borrowed)*, writer gbg |
| `dfb::dbeta` (`c_17`) | `beta_grad_has_value` | compute gbg *(borrowed)*, writer gbg |
| `dfb::gamma` (`c_6`) | `gamma_has_value` | reader ig small+large, compute ig small+large *(borrowed)* |
| `dfb::mask_h_w` (`c_7`) | `do_mask_h \|\| do_mask_w` | reader ig small+large, compute ig small+large *(borrowed)* |
| `tensor::gamma_grad` | `gamma_grad_has_value` | writer gbg |
| `tensor::beta_grad` | `beta_grad_has_value` | writer gbg |
| `tensor::gamma` | `gamma_has_value` | reader ig small+large |

Two specifics:

1. **`cb_out_init` is a parse-time ternary** in the borrowed compute kernel
   (`moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:73`):
   ```cpp
   constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;
   ```
   Both operands resolve regardless of the constant condition. This op is where the absent-`dgamma`
   configuration is reached most directly (you allocate `c_16`/`c_17` conditionally; the sibling
   allocates both unconditionally), so gate the ternary itself, and make sure **both** binders emit
   consistent defines for the shared kernel.

2. **The optional tensors have no always-bind fallback.** When `gamma_grad` is absent there is nothing
   to bind — the factory currently pushes a literal `0u` and hands `TensorAccessorArgs(nullptr)`.
   `#ifdef` is mandatory here, not preferred.

**Hardware config / opt_level** (silent-perf settings, no test net):
- ⚠ **Compute config is Style B** — both factories set `compute_desc.config = ComputeConfigDescriptor{}`,
  i.e. **all defaults**, with no `math_fidelity` / `fp32_dest_acc_en` / `dst_full_sync_en` /
  `math_approx_mode`. Build a `ComputeGen1Config` **directly** and let its defaults stand. **Do not
  reroute through `ttnn::to_compute_hardware_config`** — the TTNN helper's defaults are the
  high-performance ones and would silently flip every field this op leaves alone. (This is the one place
  this op differs materially from `moreh_layer_norm_backward`, which is Style A. Do not copy the sibling
  factory's config code.)
- **`unpack_modes`:** the legacy config sets no `unpack_to_dest_mode`, so there is nothing to carry
  over. And with `enable_32_bit_dest` left at its default `false`, the newly-*required* FP32 entry rule
  does not fire either. Expect an empty `unpack_modes` — but confirm against the resolved config rather
  than assuming.
- DM kernels use plain `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` → the arch-agnostic
  `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
- `opt_level`: `grep -n opt_level` returns nothing in either factory, so every kernel is at its legacy
  default — DM `O2` (matches Metal 2.0), **compute `O3`**. Set
  `compiler_options.opt_level = KernelBuildOptLevel::O3` explicitly on **all four** compute `KernelSpec`s
  (two factories × two core groups).

**Preserved multiplicity:** both factories build **two compute `KernelDescriptor`s** over disjoint core
groups with different per-group CTAs (`num_channels_per_core_group_1/2`,
`num_rows_per_core_group_1/2`). Keep two `KernelSpec`s in two `WorkUnitSpec`s. Do **not** demote the
per-group count to an RTA.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** **this op borrows all three of its compute kernels** from
  `moreh_layer_norm_backward`:
  - `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`
  - `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp`
  - `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp`

  No `_metal2` fork exists for any of them. The census (`grep -rl <filename>
  ttnn/cpp/ttnn/operations/`) found **exactly two consumers each — this op and its owner, and nothing
  else.**

  Because the invoker assigned **both ops in one branch and PR**, this is the bundled-port case: convert
  the three kernels **in place**, in their owner's directory, and port both binders in the same change.
  Confirm that assignment before you touch them — a consumer list is not by itself authorization. If the
  plan changes and only this op lands, create a `_metal2` fork **beside each original** (the sanctioned
  write outside your own op), add the pointer comment to each original, and point your
  `KernelSpec::source` at the forks.

  **Name the bindings for the kernel, not for this op's factory locals** — whatever the conversion uses
  becomes the interface both ops inherit. Two names to concede:
  - `c_4` — your factory calls it `one`; the kernel says `cb_scaler` → **`scaler`**.
  - `c_5` (input_grad) — your factory calls it `inner_size(==n)`; the kernel says `cb_n_recip_n` →
    **`n_recip_n`**.

  **You are the only binder of `c_6` (mask_w) on the shared gamma_beta_grad compute kernel.**
  `moreh_layer_norm_backward` never allocates it (its `do_mask_w` is compile-time false because
  `is_groupnorm == false`). So that `#ifdef` branch is live on your side and dead on theirs — do not let
  the other port delete it.

- **RTA varargs:** none — but this op is where the *trap* shape lives. Every dataflow kernel walks its
  runtime args with a running `get_arg_val<uint32_t>(i++)` counter at the top of `kernel_main`
  (`reader_...gamma_beta_grad.cpp:13-25`, `writer_...gamma_beta_grad.cpp:13-19`,
  `reader_...input_grad_{small,large}.cpp:13-26`, `writer_...input_grad.cpp:13-16`). **A running
  `arg_index++` is not a vararg signal.** These are distinct fields, read once each, in a fixed block,
  before any loop → **all named**. The counter itself disappears.

  Same for the CB ids: these kernels assign them with a running `uint32_t cb_id{0}; const auto
  cb_id_x = cb_id++;` counter (`reader_...gamma_beta_grad.cpp:33-40`,
  `reader_...input_grad_small.cpp:35-43`). The whole counter block deletes — each becomes a `dfb::name`
  token.

- **`get_tile_size(cb_id)` → `dfb.get_tile_size()`** (whitelist rule 7) at 19 sites across your five
  dataflow kernels: `reader_...gamma_beta_grad.cpp:99-102` · `writer_...gamma_beta_grad.cpp:33,37` ·
  `reader_...input_grad_small.cpp:78,88,108-111` · `reader_...input_grad_large.cpp:78,88,108-111` ·
  `writer_...input_grad.cpp:30`. The Device 2.0 gate sanctions the free function; the Metal 2.0 port
  still moves it onto the object. Note `reader_...input_grad_{small,large}.cpp:78` passes it *into* the
  donor helper — `generate_mask_h_w(dfb_mask_h_w, mask_h, mask_w, get_tile_size(cb_id_mask_h_w))`
  becomes `…, dfb_mask_h_w.get_tile_size())`.

- **Three pre-existing oddities to preserve, not fix** (all written up in the audit's Misc anomalies):
  the all-default `ComputeConfigDescriptor{}` despite the op carrying a `compute_kernel_config`
  attribute; `num_groups` / `num_channels` / `origin_h` / `origin_w` / `num_inner_tiles` riding as
  per-core RTAs despite being cache-key-invariant; and the literal-`0u` absent-optional sentinel where
  the sibling op uses a null `Buffer*`. Port them as-is.
