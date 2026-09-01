# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Port scope: ONE factory — `MatmulMultiCoreReuseOptimizedProgramFactory`.** The op has eight
factories across two DeviceOperations; only this one was audited and only this one is cleared. Do not
widen.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no site)

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers` *(carry this line into the port report's Provenance section)*

> ## ⚠ Resolve this before you convert anything
>
> **This factory's pybound `create_descriptor` has a live, in-tree Python consumer**, and its
> `core_range_set` parameter is that consumer's interface — not the vestigial hook artifact the
> recipe's exception 2 describes.
>
> - `models/experimental/ops/descriptors/matmul.py:120` calls
>   `factory.create_descriptor(operation_params, tensor_args, [out], core_range_set)` on whichever
>   factory `ttnn.matmul_select_program_factory(...)` returns (`:98`). It excludes only
>   `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (`:28`, `:39`) — **this factory is
>   selectable.**
> - That file is reached from `models/experimental/ops/descriptors/__init__.py` and used by
>   `op_descriptor.py:138` and `fusion/fusion.py:871,879`.
> - The class is **exported into the public `ttnn` namespace** (`ttnn/__init__.py:541`, aliased at
>   `ttnn/ttnn/operations/matmul.py:25`) — the only matmul *factory* so exported.
>
> The readiness sheet marks `Pybind descriptor = PR`, so removal may already be in hand in an
> in-flight PR. **Confirm that before porting rather than assuming it.** The audit raised this as its
> first question; if you were handed this brief without an answer, ask.

**Shape of the work.** The factory itself is among the simplest in the op: no multicast, **zero
semaphores**, one program builder, two DM kernels of which both are private. The substance is the
pybind question above, two compute self-loops, an aliased pair, and a threshold you must not
"correct".

## TTNN factory analysis

- **Current concept:** `descriptor` — `create_descriptor` returning a `ProgramDescriptor`
  (`…reuse_optimized_program_factory.hpp:13-17`).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (base). The factory has **no**
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and you write
  one method. **Do not add an override.**
- **Custom `compute_program_hash`:** none framework-visible. A `compute_descriptor_program_hash`
  helper at `device/matmul_device_operation.hpp:50` is *deliberately* not named
  `compute_program_hash` and is reached only through a pybind alias — which the Python descriptor
  framework does call (`matmul.py:105`). **Leave all of it alone.**
- **Gate-cleared, confirmed absent:** a non-`none` `TensorParameter relaxation` ·
  `get_dynamic_runtime_args` · smuggled pointers (verified — the factory has **no** `.address()`
  expression at all).

## Construct — to do

### Tensor bindings — four, all Case 1

`in0`, `in1`, `output`, and (conditionally) `bias`. Each is pushed into `emplace_runtime_args` as a
tensor object (`:459` in0; `:464`, `:467`, `:470` in1 / output / bias) and consumed kernel-side
through a `TensorAccessor`. Straight translation to `TensorParameter` / `TensorBinding`; the address
slots and their `TensorAccessorArgs` plumbing both disappear, so re-index what remains. No Case 2
site — the `get_bank_base_address` bridge is not needed anywhere.

### CB endpoints — six CBs, and **zero semaphores**

There is no multicast here, so each core reads its own operand blocks. Nothing to translate on the
semaphore side at all.

| CB | Disposition |
|---|---|
| `c_0` in0 | plain 1:1 — in0 reader PRODUCER, compute CONSUMER. `borrowed_from` in0 when in0 is sharded |
| `c_1` in1 | plain 1:1 — in1 reader/writer PRODUCER, compute CONSUMER. `borrowed_from` in1 when in1 is sharded |
| `c_3` bias | plain 1:1 — in1 kernel PRODUCER, compute CONSUMER; **conditional binding** |
| `c_4` out | plain 1:1 — compute PRODUCER, in1 kernel CONSUMER. `borrowed_from` output when output is sharded |
| `c_5` interm0 | **compute self-loop** — bind compute both PRODUCER and CONSUMER |
| `c_10` in0 transposed | **compute self-loop**, **conditionally bound** on `in0_transpose_tile` |

Both self-loops are genuine FIFO users (compute writes and reads back), not sync-free — do not
convert either to a scratchpad.

### Aliased DFBs — two members, config-dependent

When `interm0_data_format == output_data_format` and not (`untilize_out` with more than one W
subblock), a **single `CBDescriptor` carries both `c_4` and `c_5`** (`:614-624`), with
`output_cb_desc.tensor = output_is_sharded ? &output : nullptr`. Port as **two
`DataflowBufferSpec`s with mutual `advanced_options.alias_with`** (same total size, same bound
kernels, strict clique). In the other branch (`:598-607`) they are separate descriptors and no
aliasing applies. **Derive per instantiation.** Exactly two members — no third alias index.

### ⚠ `packer_l1_acc_en` uses `> 2` here, not `> 1` — carry it verbatim

Line 112: `bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);`

Every other matmul factory uses `> 1`. This is not a typo to fix. It feeds `interm0_data_format`
(`:114`), which decides whether the aliased-DFB branch above is taken at all — so "correcting" it
would change both the intermediate format and the CB topology. That is exactly the silent behaviour
change the porting invariant forbids. If you think it is a bug, it goes in the report, not the diff.

### Preserved multiplicity — two compute KernelSpecs

The factory runs `split_work_to_cores` and emits two `ComputeConfigDescriptor`s of the same source,
for `core_group_1` and `core_group_2` (`:500-504`, `:543-547`), the second conditional on the group
being non-empty. Port as **two KernelSpecs of the same source in two WorkUnitSpecs over disjoint node
sets**, both binding the same DFBs with the same roles. Each node sees exactly one instance, so these
are ordinary single-role bindings — **not** `allow_instance_multi_binding`. Moving the per-group
count to an RTA to collapse them is the documented anti-pattern.

### `opt_level` — set `O3` explicitly on **both** compute KernelSpecs

`grep -n opt_level` returns nothing. An absent `KernelDescriptor::opt_level` still resolves to the
legacy per-kernel-type default — **`O3` for a `ComputeConfigDescriptor`** — while Metal 2.0's
`CompilerOptions` defaults to `O2`. There are two compute specs here; each needs its own.

### Hardware config — Style A, nothing dropped

The factory resolves a TTNN `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:78-79`),
and **both** compute descriptors set all four helper-covered knobs (`:500-504`, `:543-547`). So
`to_compute_hardware_config(device->arch(), config)` translates faithfully with nothing to reapply by
hand. `packer_l1_acc` is genuinely consumed (`:112`, above) but has no Metal 2.0 counterpart.

**No `unpack_to_dest_mode` is set**, so there is no legacy table to reindex. An `unpack_modes` entry
is required only under the Float32-DFB-with-`enable_32_bit_dest` rule — check the resolved config,
not the tensor dtypes.

Both DM kernels use plain `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`, so the
arch-agnostic `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`
helpers apply. **Match on the resolved triple, not the role name** —
`reader_writer_bmm_tile_layout_in1` is configured as a *writer* even though it also reads.

### Device-op-class edits this port forces

Both are complicated by the Python consumer above — read that box first.

1. **Delete the pybound `create_descriptor`** (`matmul_nanobind.cpp:1242-1254`).
2. **Drop the `core_range_set` parameter** — but note that **unlike the sibling factories, this one
   genuinely uses it**: `:219` `else if (core_range_set.has_value())` feeding `split_work_to_cores`
   at `:227`. Production C++ always passes `std::nullopt`, so the branch falls through to the
   `program_config.allowed_worker_cores` path. Dropping the parameter therefore also means deleting
   that live branch — a larger edit than the equivalent elsewhere.

**A second bound member needs a decision, not a default.** The `nb::class_` block binds
`default_core_range` too (`:1255-1258`). It does **not** reference the vanished `create_descriptor`,
so the sanctioned exception does not cover it — but it has **no production C++ caller at all**
(only its declaration at `hpp:19`, its definition at `cpp:31`, and the pybind), and it exists purely
to let Python compute a core range for `create_descriptor`. Leaving it orphaned is defensible;
removing it exceeds the exception. **Flag it; do not decide it.**

Exception 3 does not apply — the op has a proper `program_factory_t` variant.

## Watch for

- **Only one kernel is shared — but it needs the first fork.**

  | Kernel | Binders | Rung |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0.cpp` | **1** (this factory) | **convert in place** |
  | `dataflow/reader_writer_bmm_tile_layout_in1.cpp` | **1** (this factory) | **convert in place** |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **6** | **rung 2 — create the fork** |

  The compute kernel's other five binders are `matmul_multicore_reuse_batched_hs_dram_sharded`,
  `matmul_multicore_reuse_mcast_dram_sharded`, `matmul_multicore_reuse_mcast_2d`,
  `matmul_multicore_reuse_mcast_1d` (a file hosting **two** factories), and the sparse device-op's
  factory. The rung-1 check was run locationally: `find` over `matmul/device/kernels/` returns
  **zero** `*_metal2*` files. **Name the fork's bindings for the kernel, not for this factory** —
  those names become the interface all five other consumers inherit and cannot change. The audit
  raised this as a question worth settling centrally; check for an answer before picking names.

- **Two DM kernels, not three — and the "writer" also reads.**
  `reader_writer_bmm_tile_layout_in1.cpp` reads in1 **and** writes the output from the writer
  processor, leaving the reader processor to handle in0 only. Expect DFB roles that do not line up
  with kernel names, and pick the hardware-config helper by resolved triple rather than by role.

- **`transpose_a` / `transpose_b` are a live configuration axis.** They flip the in0/in1 stride
  computations (`:267-272`) and gate the `c_10` transposed-in0 CB and its `in0_transpose_tile` CTA.
  Map the DFB set and roles per transpose setting — a topology derived with transpose off will miss
  `c_10` entirely.

- **The bias guard lives inside the factory body — carry it verbatim.** The full-block bias path sets
  `BIAS_FULL_BLOCK` and is protected by a `TT_FATAL` requiring `N == per_core_N` and
  `M == per_core_M_per_batch`. That is inside your writeable surface, so it is easy to lose in a
  rewrite; the `TT_FATAL` census in the self-audit will catch it if you do, but do not loosen it.

- **RTA varargs: none — name every argument.** Both DM kernels read args through a running
  `rt_args_idx++` counter (in0 reader `:17-20`, in1 reader/writer `:17-18` onward). **These are all
  named args.** Each is a distinct field read once in a block at the top; a running counter is not a
  vararg signal, and no kernel here reads in a loop or at a data-computed index. This is the silent
  trap the recipe warns about.

- **Locate the tests and confirm the set with your invoker.** This factory is selected via
  `MatmulMultiCoreReuseProgramConfig`. `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:2706`
  exercises it by name ("MatmulMultiCoreReuseOptimizedProgramFactory via ReuseProgramConfig"), and the
  descriptor framework under `models/experimental/ops/descriptors/` reaches it too — so coverage spans
  more than the obvious op unit tests. Search broadly rather than assuming a mirror path.
