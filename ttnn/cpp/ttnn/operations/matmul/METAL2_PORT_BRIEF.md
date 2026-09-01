# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Port scope: ONE factory — `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`.** The op has
eight factories across two DeviceOperations; only this one was audited and only this one is cleared.
Do not widen.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no accessor exists)

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers` *(carry this line into the port report's Provenance section)*

**This is a cleared but not-trivial port.** Four items below carry real work, and two of them fail
quietly if mishandled: every tensor binding is Case 2, two CBs must be dropped, the compute kernel
needs a fork, and two different aliasing patterns land on the same CB pair.

## TTNN factory analysis

- **Current concept:** `descriptor` — `create_descriptor` returning a `ProgramDescriptor` is the
  factory's only member (`…batched_hs_dram_sharded_program_factory.hpp:14`).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (base). The factory has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and you write
  one method. Do **not** add an override.
- **Custom `compute_program_hash`:** none framework-visible — the op uses the default reflection
  hash. You will find a `compute_descriptor_program_hash` helper at
  `device/matmul_device_operation.hpp:50` with a comment explaining it is *deliberately* not named
  `compute_program_hash`, plus a pybind exposing it under that name. **Leave all of it alone.**
- **Gate-cleared, confirmed absent:** a non-`none` `TensorParameter relaxation` ·
  `get_dynamic_runtime_args`. A pybound `create_descriptor` **is** present — it does not gate, and
  removing it is port work (below).

## Construct — to do

### Tensor bindings — four, and all four are Case 2

No kernel in this factory constructs a `TensorAccessor` at all. Every binding delivers a base
address that the kernel then uses **raw**, with its own arithmetic:

| Binding | Host site | Kernel use | Case |
|---|---|---|---|
| `in0` | factory:563 | in0 reader RTA 3 → `{.noc_x, .noc_y, .addr = input_shard_l1_addr + read_offset}` | **2** |
| `in1` | factory:569 | in1 kernel RTA 1 → `{.bank_id, .addr = in1_tensor_addr + …}` | **2** |
| `bias` (optional) | factory:571 | in1 kernel RTA 2 → `{.bank_id, .addr = in3_tensor_addr}` | **2** |
| `output` | factory:579 | in1 kernel RTA 7 → `{.noc_x, .noc_y, .addr = output_shard_l1_addr + out_batch_offset}` | **2** |

Express each as a `TensorParameter` / `TensorBinding`, pull the base kernel-side via the sanctioned
`TensorAccessor::get_bank_base_address()` bridge, and **leave the existing raw arithmetic exactly as
it is** — do not rewrite it into accessor iteration. The addresses currently arrive as tensor objects
pushed into the runtime-arg lists, so they disappear from the RTA lists entirely; re-index what
remains.

All four sit in **data-movement** kernels, so the bridge is available and none is blocked. (A Case 2
binding in a *compute* kernel has no bridge and would block the port — none of these is.)

> **Watch-out the recipe does not spell out.** `in0` and `output` are **L1-sharded** tensors whose
> base feeds a `{.noc_x, .noc_y, .addr}` endpoint, not a bank-relative access — the Case 2 remedy is
> written against the DRAM/bank shape. The factory asserts input-storage, output-storage and worker
> core orderings are element-wise identical (factory:104-113), so the "remote" core is the worker's
> own and the address is local. Confirm `get_bank_base_address()` returns what those call sites need
> before assuming it; if it does not, stop and report rather than threading the address through an
> RTA.

### CB endpoints — seven CBs, four different dispositions

| CB | Disposition |
|---|---|
| `c_0` in0 | plain 1:1 — in0 reader PRODUCER, compute CONSUMER |
| `c_1` in1 | plain 1:1 — in1 kernel PRODUCER, compute CONSUMER |
| `c_3` bias | plain 1:1 — in1 kernel PRODUCER, compute CONSUMER. **Conditional**: bound only when bias is present, so use the conditional-binding pattern |
| `c_4` out | plain 1:1 — compute PRODUCER, in1 kernel CONSUMER |
| `c_5` interm0 | **compute self-loop** — bind compute as **both** PRODUCER and CONSUMER |
| `c_2` in2 | **dead CB — drop** (see below) |
| `c_6` out reshard | **dead CB — drop** (see below) |

**`c_5` is a self-loop, not sync-free.** Compute runs genuine FIFO machinery against it — it packs
into it and reads it back for accumulation and the bias add. A self-loop is a statement about
endpoints, not about synchronisation; do not convert it to a scratchpad.

**`c_2` and `c_6` have zero endpoints and must be dropped.** Both are borrowed views
(`cb_desc.tensor = &in0_tensor` at factory:247-259; `&out_tensor` at factory:319-331) that no kernel
binds — no kernel body references them and no named CTA carries their index. The kernels reach the
same memory by explicit NOC address from a runtime arg instead. A DFB with no producer and no
consumer is rejected by the spec validator, so there is no way to carry them across. Record each
drop with `file:line` in the port report.

> ⚠ **A question on these two is open with the op owner** (see the audit's Questions section). The
> census is unambiguous, but the borrowed-CB-over-a-resident-tensor idiom normally exists for a
> reason and the audit could not identify this one's. **Confirm before the drop lands** — the
> validator catches a dead DFB you wrongly keep, but nothing catches a live one you wrongly drop.

### Two aliasing patterns, on the same CB pair — do not conflate them

The recipe flags this distinction as correctness-critical, and this factory has one of each a few
hundred lines apart:

- **Aliased DFBs (host side, and config-dependent).** When
  `interm0_data_format == output_data_format` (factory:301-317) a **single `CBDescriptor` carries two
  `format_descriptors`**, `c_4` and `c_5` — two distinct buffers sharing one L1 region. Port as **two
  `DataflowBufferSpec`s with mutual `advanced_options.alias_with`** (same total size, same bound
  kernels, strict clique). In the other branch (factory:277-300) they are separate descriptors and no
  aliasing applies.
- **Same-FIFO aliasing (kernel side).** The compute kernel does
  `constexpr uint32_t mm_out_dfb_id = mm_partials_dfb_id;` (compute kernel line 228) — **one** CB
  under two names, sharing one set of FIFO pointers. Port as **one binding plus a `constexpr` handle
  alias**. Do **not** add a second `DFBBinding`, and do **not** model it with `alias_with` — that
  would give you two independent FIFOs at one address and silently break pointer coherence.

### Named CTAs: two to drop, one that needs `#ifdef` gating

- **Drop `cb_in0_intermediate` (`c_8`) and `cb_in1_intermediate` (`c_9`).** The factory passes both
  to the compute kernel (factory:493-494); the compute kernel references neither, under any
  `#ifdef`, and no `CBDescriptor` allocates them. Dead on both ends.
- **`cb_in0_transposed` (`c_10`) cannot simply be dropped.** Compute line 200 selects its in0 handle
  with a **parse-time ternary**:
  `in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed") : get_named_compile_time_arg_val("cb_in0")`.
  This factory always passes `in0_transpose_tile = 0` (positional CTA 17, hardcoded `0u` at
  factory:415-433), so the branch is never taken — but **both operands are name-looked-up anyway**,
  which is why the CTA exists today with no CB behind it. In Metal 2.0 a named CB index becomes a
  `DFBBinding`, and there is no DFB to bind. Use the **conditional-binding pattern**: emit a define
  from `KernelSpec::compiler_options.defines` and `#ifdef`-gate the alias so the unused branch never
  reaches name lookup.

### `opt_level` — set `O3` explicitly on the compute KernelSpec

`grep -n opt_level` on the factory returns nothing. An absent `KernelDescriptor::opt_level` still
resolves to the legacy per-kernel-type default — **`O3` for a `ComputeConfigDescriptor`** — while
Metal 2.0's `CompilerOptions` defaults to `O2` for both kinds. Leaving it unset silently drops a
level. The two DM kernels need nothing.

### Hardware config — Style A, no dropped field

The factory resolves a TTNN `ComputeKernelConfig` via `get_compute_kernel_config_args`
(factory:673-674), so translate with `to_compute_hardware_config(device->arch(), config)`. All four
helper-covered knobs are set on the compute descriptor (factory:513-516), so there is no
resolved-but-unset field to reapply. `packer_l1_acc` is genuinely consumed — it derives
`packer_l1_acc_en` and hence `interm0_data_format` (factory:161-165) — but has no Metal 2.0
counterpart, so no action.

`unpack_modes` needs an entry **only if** `interm0` resolves to `Float32` with `enable_32_bit_dest`
on. That happens when `fp32_dest_acc_en` is set, and the trigger is the **DFB's format, not the
tensor dtypes** — so read the resolved config, don't infer from the op's inputs.

Both DM kernels use **explicit custom configs** — `RISCV_1` for the in0 reader and `RISCV_0` with
`in1_noc` for the in1 kernel (factory:459-476). Replicate each field verbatim with a
`DataMovementGen1Config`; do **not** reach for `create_reader_datamovement_config` /
`create_writer_datamovement_config`, which would silently substitute the default triple.

### Device-op-class edits this port forces

Two sanctioned exceptions, both recorded under Handoff points:

1. **Delete the pybound factory entry point.** `matmul_nanobind.cpp:1325-1338` is an
   `nb::class_<…BatchedHSDRAMShardedProgramFactory>` block whose only member is `create_descriptor`.
   That method vanishes, so the block goes. Leave the separate `nb::class_<MatmulDeviceOperation>`
   block at lines 1222-1237 untouched — it binds device-op methods that survive.
2. **Drop the pybind-hook-only parameter.** `create_descriptor`'s fourth argument,
   `const std::optional<CoreRangeSet>& core_range_set`, is **ignored by the factory body** — spelled
   `/*core_range_set*/` at `…batched_hs_dram_sharded_program_factory.cpp:603`. Drop it; there is no
   production default to inline.

Exception 3 does not apply — the op has a proper `program_factory_t` variant.

## Watch for

- **Shared kernel — the compute kernel needs a fork, and you are the first to reach it.**

  | Kernel | Binders | Rung |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | **1** (this factory) | convert in place |
  | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | **1** (this factory) | convert in place |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **6** | **rung 2 — create the fork** |

  The compute kernel is bound by this factory plus `matmul_multicore_reuse_optimized`,
  `matmul_multicore_reuse_mcast_dram_sharded`, `matmul_multicore_reuse_mcast_2d`,
  `matmul_multicore_reuse_mcast_1d`, and the sparse device-op's
  `sparse_matmul_multicore_reuse_mcast_1d_optimized` — all binding the same path, so all are genuine
  consumers. Converting it in place breaks the five that are not porting here.

  The rung-1 check was run **locationally**: `ls` of `device/kernels/compute/` shows no
  `bmm_large_block_zm_fused_bias_activation_metal2.cpp`. So **create it beside the original**,
  convert the copy, point this factory's `KernelSpec::source` at it, leave the original untouched
  apart from the pointer comment, and record the five remaining consumers in the port report as the
  sunset list. That list is **coordination, not authorization** to convert in place.

  **Name the fork's bindings for the kernel, not for this factory** — whatever names you choose
  become the interface the other five inherit, and they will not be able to rename them.

  The fork's `#include` closure pulls in `bmm_fused_activation.hpp`, an in-op sibling in the same
  directory. It exposes **no** CB-id parameters, so it needs no conversion and no fork of its own;
  the forked kernel keeps including it.

- **RTA varargs: none — name every argument.** Both DM kernels read their args as distinct fields at
  constant indices in a block at the top (in0 reader slots 0-3; in1 kernel slots 0-7, with slot 2
  gated by `FUSE_BIAS`). Nothing here justifies `get_vararg`.

- **Sanctioned free functions with two different port forms.** The in1 kernel declares
  `constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb_id_in1);` (lines 59-60) — the
  `constexpr` declaration is the whole test, so these **keep the free-function form with the binding
  token**: `get_tile_size(dfb::in1)`. Do not move them onto the object and do not demote them to
  `const` to make a member getter fit. Contrast line 113 in the same file,
  `dfb_in3.get_tile_size()`, which is already the member form in a non-`constexpr` context and stays
  as it is.

- **One `get_local_cb_interface` site, in dead code, in the shared kernel.** Compute line 116 reads
  `get_local_cb_interface(mm_partials_dfb_id).fifo_rd_ptr` — a cursor *read*, which maps to the DFB's
  public `get_read_ptr()` peek (not an `evil_set_*`). It sits behind
  `if (mm_partials_reload_dfb_id != mm_partials_dfb_id)` (line 112), a condition the caller makes
  always-false by construction (line 212). Convert it in the fork; do not delete the branch.

- **Storage cores and worker cores are the same cores.** The factory asserts all three orderings
  match element-wise (factory:104-113), so the NOC reads and writes that look remote are same-core.
  Keep that in mind when reasoning about the `in0` / `output` bindings above.

- **Locate the tests and confirm the set with your invoker.** This factory is selected only by an
  explicitly-constructed `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig` — the
  auto-config path never produces one — so its coverage will not appear under a generic matmul
  filter. Search broadly rather than assuming a mirror path.
