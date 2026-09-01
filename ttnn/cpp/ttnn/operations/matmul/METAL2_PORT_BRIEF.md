# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Port scope: ONE factory — `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`.** The op has
eight factories across two DeviceOperations; only this one was audited and only this one is cleared.
Do not widen.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no accessor exists)

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers` *(carry this line into the port report's Provenance section)*

**Shape of the work.** This is a comparatively clean port: the **base** concept (no override to
translate), one program builder, three kernels of which only one is shared, and no dropped
compute-config field. The substance is in the CB dispositions — two sync-free borrowed CBs, a compute
self-loop, and a config-dependent aliased pair — plus the absent `opt_level`.

## TTNN factory analysis

- **Current concept:** `descriptor` — `create_descriptor` returning a `ProgramDescriptor` is the
  struct's **only** member (`…mcast_dram_sharded_program_factory.hpp:14-18`).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (base). The factory has **no**
  `override_runtime_arguments` anywhere, so the framework refreshes tensor bindings on cache hit and
  you write one method. **Do not add an override** — doing so would move the op to the custom concept
  and make you responsible for the entire cache-hit refresh.
- **Custom `compute_program_hash`:** none framework-visible — the op uses the default reflection
  hash. You will find a `compute_descriptor_program_hash` helper at
  `device/matmul_device_operation.hpp:50` that is *deliberately* not named `compute_program_hash`,
  plus a pybind exposing it under that name. **Leave all of it alone.**
- **Gate-cleared, confirmed absent:** a non-`none` `TensorParameter relaxation` ·
  `get_dynamic_runtime_args` · smuggled pointers (verified — both address slots are rebound). A
  pybound `create_descriptor` **is** present; removing it is port work.

## Construct — to do

### Tensor bindings — four, split two ways

| Binding | Host delivery | Kernel use | Classification |
|---|---|---|---|
| `in1` | RTA slot [1], rebound to the tensor at `:917` | **raw** — `{.bank_id, .addr = in1_tensor_addr + …}` (in1 kernel `:87`, `:100`, `:132`, `:137`, `:148`) | **Case 2** |
| `bias` | RTA slot [2], rebound at `:919` (bias present only) | **raw** — `{.bank_id, .addr = in3_tensor_addr}` (in1 kernel `:191`) | **Case 2** |
| `in0` | **borrowed CB** `c_2` (`cb_desc.tensor = &in0_tensor`, `:576`) | `dfb_in2.get_read_ptr()` (in0 kernel `:71`) | **clean** — borrowed DFB |
| `output` | **borrowed CB** `c_6` (`cb_desc.tensor = &out_tensor`, `:633`) | `dfb_out_reshard.get_write_ptr()` (in1 kernel `:218`) | **clean** — borrowed DFB |

For the two **Case 2** bindings: express each as a `TensorParameter` / `TensorBinding`, pull the base
kernel-side via the sanctioned `TensorAccessor::get_bank_base_address()` bridge, and **leave the
existing raw arithmetic exactly as it is** — do not rewrite it into accessor iteration. Both sit in
**data-movement** kernels, so the bridge is available and neither is blocked. (A Case 2 binding in a
*compute* kernel has no bridge and would block the port; neither of these is.)

For the two **clean** bindings: the borrowed DFB *is* the tensor access. Express each as
`DataflowBufferSpec::borrowed_from` naming the corresponding `TensorParameter`. Do **not** force them
into Case 1 or Case 2.

### CB endpoints — seven CBs, four dispositions

There is a single code path (the three `skip_*` flags are hardcoded off — see Watch for), so no
per-config census flip. The only conditionals are bias presence and the output/intermediate format
branch.

| CB | Disposition |
|---|---|
| `c_0` in0 mcast dest | plain 1:1 — in0 kernel PRODUCER, compute CONSUMER |
| `c_1` in1 | plain 1:1 — in1 kernel PRODUCER, compute CONSUMER |
| `c_3` bias | plain 1:1 — in1 kernel PRODUCER, compute CONSUMER; **conditional binding** (bias present) |
| `c_4` out | plain 1:1 — compute PRODUCER, in1 kernel CONSUMER |
| `c_5` interm0 | **compute self-loop** — bind compute both PRODUCER and CONSUMER |
| `c_2` in0 sharded (borrowed) | **self-loop** — sync-free, one toucher |
| `c_6` out reshard (borrowed) | **self-loop** — sync-free, one toucher |

**`c_2` and `c_6` are the pair to get right.** Each is a borrowed view that exactly one kernel reaches
by base pointer with **no `reserve_back` / `push_back` / `wait_front` / `pop_front` anywhere**. Bind
the touching kernel as both PRODUCER and CONSUMER; the kernel code is untouched and runtime behaviour
is identical to the legacy CB. Both are genuinely kernel-referenced — `c_2` reaches the in0 kernel as
the named CTA `cb_in0_sharded` (`:436`), `c_6` reaches the in1 kernel as `cb_out_reshard` (`:454`) —
so neither is a dead CB.

> Both are also `LocalTensorAccessor` candidates for the **post-port sync-free style pass**
> (sync-free *and* borrowed is precisely that pass's target). That is a separate pass with its own
> procedure — **do not do it here.** Note them in the port report for whoever runs it.

**`c_5` is a self-loop, not sync-free.** Compute runs genuine FIFO machinery against it — it packs
into it and reads it back. A self-loop is a statement about endpoints, not about synchronisation; do
not convert it to a scratchpad.

### Aliased DFBs — a two-member group, config-dependent

When `interm0_data_format == output_data_format` and not (`untilize_out` with more than one W
subblock), a **single `CBDescriptor` carries both `c_4` and `c_5`** (`:605-620`) — two distinct
buffers sharing one L1 region. Port as **two `DataflowBufferSpec`s with mutual
`advanced_options.alias_with`** (same total size, same bound kernels, strict clique). In the other
branch (`:581-604`) they are separate descriptors and no aliasing applies. **Derive per
instantiation.** The group here is exactly two members — there is no third alias index.

### Named CTAs: two to drop, one that needs `#ifdef` gating

- **Drop `cb_in0_intermediate` (`c_8`) and `cb_in1_intermediate` (`c_9`)** (`:507-508`). The compute
  kernel references neither, under any `#ifdef`, and no `CBDescriptor` allocates them.
- **`cb_in0_transposed` (`c_10`, `:509`) cannot simply be dropped.** Compute line 200 selects its in0
  handle with a **parse-time ternary**
  (`in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed") : …("cb_in0")`), so both
  operands are name-looked-up regardless of the condition. There is no CB behind `c_10`, and in
  Metal 2.0 a named CB index becomes a `DFBBinding`. Use the **conditional-binding pattern**: emit a
  define from `KernelSpec::compiler_options.defines` and `#ifdef`-gate the alias.

### `opt_level` — set `O3` explicitly on the compute KernelSpec

`grep -n opt_level` returns nothing. An absent `KernelDescriptor::opt_level` still resolves to the
legacy per-kernel-type default — **`O3` for a `ComputeConfigDescriptor`** — while Metal 2.0's
`CompilerOptions` defaults to `O2` for both kinds. Silent perf loss with no test net. The two DM
kernels need nothing.

### Hardware config — Style A, and nothing dropped

The factory resolves a TTNN `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:1041`); the
builder **receives** `dst_full_sync_en` (signature `:51`) and **sets** it (`:526`) alongside the other
three knobs (`:523-527`). All four helper-covered fields are set, so
`to_compute_hardware_config(device->arch(), config)` translates faithfully with nothing to reapply by
hand. `packer_l1_acc` has no Metal 2.0 counterpart; no action.

**No `unpack_to_dest_mode` is set** by this factory, so there is no legacy table to reindex. The only
reason an `unpack_modes` entry would be required is the Float32-DFB-with-`enable_32_bit_dest` rule —
check the *resolved config*, not the tensor dtypes.

**DM configs are custom — replicate, don't use the helpers.** `RISCV_1 / in0_noc` (`:439`) and
`RISCV_0 / in1_noc` (`:457`), where both NOCs come from `preferred_noc_for_dram_write` /
`preferred_noc_for_dram_read` rather than the reader/writer defaults. Copy every field verbatim into
a `DataMovementGen1Config`. **`in1_noc` is load-bearing beyond perf**: multi-worker mode requires it
to be `NOC_0` (`:120`).

### Device-op-class edits this port forces

1. **Delete the pybound factory entry point.** `matmul_nanobind.cpp:1309-1323` is an
   `nb::class_<…MultiCastDRAMShardedProgramFactory>` block whose only member is `create_descriptor`.
   Leave the separate `nb::class_<MatmulDeviceOperation>` block at lines 1222-1237 untouched.
2. **Drop the pybind-hook-only parameter** — `create_descriptor`'s `core_range_set`, which the
   factory body ignores and which exists only for that hook.

Exception 3 does not apply — the op has a proper `program_factory_t` variant.

## Watch for

- **Only one kernel is shared — but it needs the first fork.**

  | Kernel | Binders | Rung |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | **1** (this factory) | **convert in place** |
  | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | **1** (this factory) | **convert in place** |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **6** | **rung 2 — create the fork** |

  The compute kernel's other five binders are `matmul_multicore_reuse_optimized`,
  `matmul_multicore_reuse_mcast_1d` (a file hosting **two** factories),
  `matmul_multicore_reuse_mcast_2d`, `matmul_multicore_reuse_batched_hs_dram_sharded`, and the sparse
  device-op's factory. The rung-1 check was run locationally: `find` over
  `matmul/device/kernels/` returns **zero** `*_metal2*` files, so you create the first fork.
  **Name its bindings for the kernel, not for this factory** — those names become the interface all
  five other consumers inherit and cannot change. The audit raised this as a question worth settling
  centrally; check whether an answer came back before you pick names.

- **`SKIP_MCAST` appears twice, twelve lines apart, meaning different things.** `:390` sets it
  **unconditionally** for the in1 sender/writer defines (alongside `OUT_SHARDED` at `:389`), while
  `:396` sets it for the in0 sender **only if** `skip_in0_mcast`. Read the target map, not the define
  name.

- **Three `skip_*` flags are hardcoded off — treat them as dead, not as config axes.**
  `create_descriptor` sets `skip_compute = false`, `skip_in0_mcast = false`, `skip_write_back = false`
  (`:1036-1038`). They gate `SKIP_COMPUTE`, the in0-side `SKIP_MCAST`, and `SKIP_WRITE_BACK`
  (`:392-399`), none of which is ever emitted on this dispatch path. **Do not census the CBs three
  extra times for them**, and do not remove them either — they are out of the port's scope.

- **`num_workers_per_dram_bank` *is* a real config axis, with hard constraints.** Values above 1 are
  **Blackhole-only** (`:117`), require the in1 DM kernel on **NOC_0** (`:120`), and need the weight
  shard width in tiles to divide by it (`:144`). It changes worker count, reader-to-bank assignment
  and `per_core_N_in1_sender` — but it adds no kernel and no CB, so the endpoint dispositions above
  hold across it. Port the field through; do not normalise it away.

- **`last_subblock_w_valid` looks redundant and is not.** The factory picks its own subblock shape,
  then runs a widening pass that can pad `per_core_N_compute` past what the reader actually pushed;
  `last_subblock_w_valid` tells compute how many lanes of the final subblock are backed by real
  tiles. Carry the CTA across faithfully.

- **RTA varargs: none — name every argument.** No kernel reads arguments in a loop, at a
  data-computed index, or through a running counter. Every read is `get_arg_val<uint32_t>(<literal>)`
  in a block at the top. Nothing here justifies `get_vararg`.

- **Dead placeholder addresses will mislead a text search.** `:796` and `:797` compute a `.address()`
  that the variant rebinding overwrites at `:917` / `:919`. Both already carry a `smuggled-rta-ok`
  marker. They are not smuggled pointers — don't "fix" them, and don't be alarmed by them.

- **Locate the tests and confirm the set with your invoker.** This factory is reachable **only** via
  an explicitly-constructed `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` — the auto-config
  path never produces one — and the readiness sheet marks it `ProgramFactory used in llama? = yes`,
  so its coverage likely spans model-level tests as well as op unit tests. A generic matmul filter
  will not reach it. Search broadly rather than assuming a mirror path.
