# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `82c2b5569eb 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port` *(carry this line into the port report's Provenance section)*

**Two factories, two DeviceOperations, one directory.** They share no kernels with each other. Port them
as two units:

| Factory | Kernel sources it binds | Atomic unit size |
|---|---|---|
| `MorehLayerNormBackwardGammaBetaGradFactory` | reader, writer, compute | 3 |
| `MorehLayerNormBackwardInputGradFactory` | reader **small\|large**, writer, compute **small\|large** | **5** — runtime-selected on `use_large_algorithm`; all five convert together |

> **Read this with `moreh_group_norm_backward`'s brief.** That op binds **three of this op's compute
> kernels**. See [Watch for → Cross-op / shared kernels](#watch-for).

## ⚠ Your atomic unit spans both ops

**The bundled port is confirmed** (invoker, 2026-08-13): the three shared compute kernels convert **in
place**, no forks. That has a consequence which overrides the recipe's usual "port one factory at a
time" default.

A shared kernel converted in place flips to Metal 2.0 **once**, and at that moment every factory binding
it must already speak Metal 2.0 bindings or the build breaks. So the atomic unit is *one shared compute
kernel + all of its binders* — which pairs each of this op's factories with its group-norm counterpart:

| Unit | Factories that must convert together | Kernel sources |
|---|---|---|
| **A — gamma_beta_grad** | `MorehLayerNormBackwardGammaBetaGradFactory` **+** `MorehGroupNormBackwardGammaBetaGradFactory` | 5 |
| **B — input_grad** | `MorehLayerNormBackwardInputGradFactory` **+** `MorehGroupNormBackwardInputGradFactory` | 8 |

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
  change in `device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp:34` and
  `device/moreh_layer_norm_backward_input_grad_device_operation.hpp:33`. No pybind cleanup — the
  nanobind file exposes only the user-facing op.

## Construct — to do

**Tensor bindings** (per binding) — **all Case 1**, all delivered today via the `Buffer*`-binding form:

*gamma_beta_grad*
- `output_grad`, `input`, `mean`, `rstd` — Case 1 → `TensorParameter` / `TensorBinding`; reader uses
  `TensorAccessor(tensor::…)`.
- `gamma_grad` — Case 1, **conditional** on `gamma_grad_has_value`.
- `beta_grad` — Case 1, **conditional** on `beta_grad_has_value`.

*input_grad*
- `output_grad`, `input`, `mean`, `rstd`, `input_grad` — Case 1.
- `gamma` — Case 1, **conditional** on `gamma_has_value`.

All the legacy plumbing that goes with them disappears: the `TensorAccessorArgs(...).append_to(...)`
chains on the host, the `TensorAccessorArgs<N>()` / `next_compile_time_args_offset()` chains in the
kernels, and the `Buffer*` RTA slots.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none — all 11 accessor sites are the 2-arg form.

**CB endpoints:**

- **Self-loop** (compute-only intermediate, one toucher — bind the compute `KernelSpec` as both
  PRODUCER and CONSUMER):
  - gamma_beta_grad: `c_24` y, `c_25` y·dy, `c_26` Add[dy], `c_27` Add[y·dy], `c_28` x−mean, `c_29` dycopy
  - input_grad **small**: `c_24` dycopy, `c_25` y, `c_26` Sum[dy], `c_27` Sum[y·dy], `c_28` recip_nrstd,
    `c_29` tmp1, `c_30` tmp2, `c_31` tmp3
  - input_grad **large**: `c_24` dycopy, `c_25` y, `c_26` Sum[dy], `c_27` Sum[y·dy], `c_28` tmp1,
    `c_29` tmp2, `c_30` tmp3 *(no `c_31` — `im7_t = 0` under the large algorithm; correct, not a dead CB)*
- **Legal 1:1** (reader→compute, or compute→writer): everything else.
- **Multi-binding flag:** **none anywhere.** Every CB has at most two touchers, each locked to one FIFO
  role. If you find yourself reaching for `allow_instance_multi_binding`, recount.
- **Dead-CB drop:** none unconditionally. Two CBs are *config-scoped* dead (`c_16`/`c_17` when the
  matching optional output is absent) — handled by conditional binding, below, not by a drop.

**Conditional bindings — this is the bulk of the port.** Nine resources are bound only on some
compile-time path, and every one is referenced *unconditionally* by the kernel today. Host binds
conditionally + emits a `compiler_options.defines` flag + kernel `#ifdef`-gates the alias and every
expression naming it:

| Resource | Condition | Kernels to gate |
|---|---|---|
| `dfb::mask_h` (`c_5`) | `do_mask_h` | reader gbg, compute gbg |
| `dfb::mask_w` (`c_6`) | **never true in this op** — LN never allocates it | compute gbg (declaration only) |
| `dfb::dgamma` (`c_16`) | `gamma_grad_has_value` | compute gbg, writer gbg |
| `dfb::dbeta` (`c_17`) | `beta_grad_has_value` | compute gbg, writer gbg |
| `dfb::gamma` (`c_6`) | `gamma_has_value` | reader ig small+large, compute ig small+large |
| `dfb::mask_h_w` (`c_7`) | `do_mask_h \|\| do_mask_w` | reader ig small+large, compute ig small+large |
| `tensor::gamma_grad` | `gamma_grad_has_value` | writer gbg |
| `tensor::beta_grad` | `beta_grad_has_value` | writer gbg |
| `tensor::gamma` | `gamma_has_value` | reader ig small+large |

Three specifics worth handling deliberately:

1. **`cb_out_init` is a parse-time ternary** —
   `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp:73`:
   ```cpp
   constexpr auto cb_out_init = gamma_grad_has_value ? cb_dgamma : cb_dbeta;
   ```
   Both operands resolve regardless of the constant condition, so gating only the *uses* leaves this
   line failing name lookup. Gate the ternary itself. Expect this to be your first compile break.

2. **The optional tensors have no always-bind fallback.** When `gamma_grad` is absent there is nothing
   to bind — the factory currently hands `TensorAccessorArgs(nullptr)` and a null `Buffer*`. `#ifdef` is
   mandatory here, not preferred.

3. **The mask gates are CTA-computed today and must be promoted to defines.** The compute kernels
   derive `do_mask_h` / `do_mask_w` from `origin_H`/`origin_W`/`is_lastdim_layernorm`/`is_groupnorm`
   rather than reading a host flag. The audit verified the host predicate and the kernel predicate agree
   in every configuration, so the promotion is a faithful translation — but **feed the define to every
   kernel that names the resource**, not just the reader (the legacy factory emits no mask define at
   all).

**Same-FIFO aliasing — one DFB, several kernel-side names.** Both input_grad compute kernels alias
`cb_tmp1`/`cb_tmp2`/`cb_tmp3` (and `cb_y`) under working names at block scope:
`cb_xmm`, `cb_dyadd`, `cb_ydy`, `cb_ydyadd`, `cb_ndy`, `cb_ndymdysum`, `cb_yydysum`, `cb_tmp4`,
and — **in the large kernel only** — `cb_recip_nrstd`.

Keep **one** `DataflowBufferSpec` and **one** `DFBBinding` per buffer; express each working name as a
handle alias (`constexpr auto cb_xmm = dfb::tmp2;`) and construct a **single** `DataflowBuffer` object.
Do **not** reach for `advanced_options.alias_with` — that models distinct buffers sharing memory and
would silently break the shared-pointer coherence these names rely on.

⚠ **`cb_recip_nrstd` is a real distinct CB (`c_28`) in the small kernel and an alias of `cb_tmp3` in the
large kernel.** Derive the DFB map **per selected source path**, not once for the factory.

**Hardware config / opt_level** (silent-perf settings, no test net):
- Both factories build compute config **Style A** — `init_device_compute_kernel_config` →
  `get_compute_kernel_config_args` → a `ComputeConfigDescriptor` with `math_fidelity`,
  `fp32_dest_acc_en`, `dst_full_sync_en`, `math_approx_mode`. Use `ttnn::to_compute_hardware_config(arch,
  compute_kernel_config)` and mind the `dst_full_sync_en → !double_buffer_dest` inversion.
- Neither factory sets `unpack_to_dest_mode`, so there is **no legacy `unpack_modes` entry to carry
  over**. But check the newly-*required* rule: if any compute kernel consumes a `Float32` DFB with
  `enable_32_bit_dest = true`, the validator demands an explicit entry. Under `fp32_dest_acc_en` the
  intermediate CBs (`c_24`+) are `tt::DataFormat::Float32` and the compute kernel consumes them, so
  **expect to add entries the legacy code did not have** — derive each from the legacy default
  (`Default` → `UnpackToSrc`), do not guess.
- DM kernels use plain `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` → the arch-agnostic
  `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
- `opt_level`: `grep -n opt_level` returns nothing in either factory, so every kernel is at its legacy
  default — DM `O2` (matches Metal 2.0), **compute `O3`**. Set
  `compiler_options.opt_level = KernelBuildOptLevel::O3` explicitly on **all four** compute `KernelSpec`s
  (two factories × two core groups).

**Preserved multiplicity:** both factories build **two compute `KernelDescriptor`s** over disjoint core
groups with different per-group CTAs (`num_cols_per_core_group_1/2`, `num_rows_per_core_group_1/2`).
Keep two `KernelSpec`s in two `WorkUnitSpec`s. Do **not** demote the per-group count to an RTA.

## Watch for

- **CB endpoints (multi-binding):** none. No hidden second writer, no multi-reader, no dual-instance
  work-split in this op.
- **Cross-op / shared kernels:** **three kernels in this op's own directory are bound by
  `moreh_group_norm_backward`:**
  - `device/kernels/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`
  - `device/kernels/moreh_layer_norm_backward_input_grad_large_kernel.cpp`
  - `device/kernels/moreh_layer_norm_backward_input_grad_small_kernel.cpp`

  No `_metal2` fork exists for any of them. The census (`grep -rl <filename>
  ttnn/cpp/ttnn/operations/`) found **exactly two consumers each — this op and
  `moreh_group_norm_backward`, and nothing else.**

  Because the invoker assigned **both ops in one branch and PR**, this is the bundled-port case: convert
  the three kernels **in place** and port both binders in the same change. Confirm that assignment
  before you touch them — a consumer list is not by itself authorization. If the plan changes and only
  one op lands, fall back to creating a `_metal2` fork beside each original.

  **Name the bindings for the kernel, not for either factory's locals.** The two ops disagree on one
  name: layer-norm's `c_4` is *scaler*, group-norm's `c_4` is *one*. The kernel says `cb_scaler`, so
  **`scaler`** is the binding name.

  The group-norm factories also bind `c_6` (mask_w) on the shared gamma_beta_grad compute kernel, which
  layer-norm never allocates. Your `#ifdef` for `mask_w` therefore has a live consumer on the other
  side — do not delete the branch as dead.

- **RTA varargs:** none. Every kernel reads its args as a block of distinct fields at the top of
  `kernel_main` → **all named**. Note the group-norm siblings use a running `get_arg_val(i++)` counter;
  that is *not* a vararg signal either. Do not smuggle any of these into varargs.

- **`get_tile_size(cb_id)` → `dfb.get_tile_size()`** (whitelist rule 7) at 13 sites across 5 kernels:
  `reader_..._gamma_beta_grad.cpp:27,158,159` · `reader_..._input_grad_small.cpp:27,167,168,169` ·
  `reader_..._input_grad_large.cpp:27,168,169,170` · `writer_..._gamma_beta_grad.cpp:35,36` ·
  `writer_..._input_grad.cpp:28`. The Device 2.0 gate sanctions the free function; the Metal 2.0 port
  still moves it onto the object.

- **`read_mean_rstd` is a file-local kernel template taking `uint32_t cb_id`.** It appears in all three
  readers. `dfb::mean` / `dfb::rstd` convert implicitly, so the call sites are fine; inside the helper,
  swap `get_tile_size(cb_id)` for the constructed object's `dfb.get_tile_size()`. It is in your
  directory — an ordinary in-scope edit.

- **Two pre-existing oddities to preserve, not fix** (both written up in the audit's Misc anomalies):
  CTA slot 3 of the input_grad compute kernels is named `Wt` but carries `num_inner`; and
  `normalized_dims` / `mean_rstd_height` / `mean_rstd_width` ride as per-core RTAs despite being
  cache-key-invariant. Port them as-is.
