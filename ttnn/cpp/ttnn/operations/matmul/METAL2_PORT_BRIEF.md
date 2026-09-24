# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/matmul` (sparse sub-tree)

> Audit cleared all gates. This is your actionable input; the full record is in
> `METAL2_PREPORT_AUDIT.md`.

**Scope: ONE ProgramFactory — `SparseMatmulMultiCoreReuseMcast1DProgramFactory`**, on
`SparseMatmulDeviceOperation` (`device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.{hpp,cpp}`).
It is the device-op's only variant alternative. `MatmulDeviceOperation`'s factories are out of scope.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓ *(never fired)*

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**One gate carried a caveat you should know about:** the readiness sheet's `Concept` column still
reads `legacy device-op` for this row, while the code is on `descriptor` as of #57369. The audit
treated that as sheet staleness rather than a *spreadsheet-is-broken* GATE, because the verdict cell
reads `yes (with PD step)` and the PD step has landed. Reasoning and the route to the sheet owner
are in the audit's *Gate detail*. Nothing for you to act on; it does not change the target concept.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — a lone
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` at
  `…mcast_1d_optimized.hpp:24`. No `cached_program_t`, no `override_runtime_arguments`.
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** — the base concept. There is no
  `override_runtime_arguments` to translate, so **skip the recipe's
  *Translating `override_runtime_arguments`* subsection entirely.**
- **Custom `compute_program_hash`:** none. The declaration and definition are both **commented
  out** (`sparse_matmul_device_operation.hpp:33`, `.cpp:504`) — the framework uses the default
  reflection hash. A grep for `compute_program_hash` *will* hit those comment lines; read the line,
  don't count the match. They are device-op-class code: **leave them exactly as they are.**
- **No device-op-class edit is forced.** Nothing under `device/sparse/` is pybound, so there is no
  `create_descriptor` pybind line to delete (exception 1 does not apply); the factory already lives
  in `program_factory_t = std::variant<…>` (`hpp:21`), so it is not the direct-descriptor shape
  (exception 3 does not apply); and it takes no pybind-hook-only parameter (exception 2 does not
  apply). Your writeable surface is the factory `.cpp` + `.hpp` and nothing else.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a
  `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`.

## Construct — to do

### Tensor bindings (per binding) — all Case 1

Every binding is fed into a `TensorAccessor` and accessed only through it; none is used raw, so
**no `get_bank_base_address` bridge is needed anywhere.**

| Binding | Legacy delivery | Kernel side |
|---|---|---|
| `in0` | `Buffer*` binding, `in0_args[0]` (`:790`) | `TensorAccessor(tensor::in0)` in the in0 sender |
| `in1` | `Buffer*` binding, `in1_args[0]` (`:872`) | `TensorAccessor(tensor::in1)` in the in1 sender/writer |
| `sparsity` | `Buffer*` binding, `in0_args[7]` (`:791`) | `TensorAccessor(tensor::sparsity)` in the **in0 sender** |
| `indices` *(indexed mode only)* | `Buffer*` binding, `in1_args[6]` (`:873`) | `TensorAccessor(tensor::sparsity)` in the **in1 sender/writer** |
| `output` | `Buffer*` binding, `in1_args[7]` (`:874`) | `TensorAccessor(tensor::out)` in the in1 sender/writer |

**⚠ The two "sparsity" slots are different tensors.** `…mcast_1d_optimized.cpp:135`:

```cpp
const Tensor& in1_sparsity_tensor = use_indices ? tensor_args.optional_input_tensors.at(0).value() : sparsity;
```

- The **in0 sender's** slot (`c_6`, CTA accessor args at `:372`) is *always* the real sparsity mask.
- The **in1 sender/writer's** slot (`c_7`, CTA accessor args at `:431`) is the **active-group id
  list** in indexed/gather mode and the mask otherwise.

So when `use_indices` is true you need **two** `TensorParameter`s, and the in1 sender binds the
*indices* one. When it is false, both kernels bind the same `TensorParameter`. Accessor names are
per-kernel-scoped, so the fork's `tensor::sparsity` works for both — you vary the
`tensor_parameter_name`, not the `accessor_name`.

Five `->address()` reads at `:771`, `:782`, `:809`, `:818`, `:822` are **dead**: they fill a staging
`uint32_t` vector whose slots the `Buffer*` assignments then overwrite. They go away with the vector.

### TensorParameter relaxation

`none`. Strict `TensorSpec` match on every parameter. Do not add a relaxation.

### TensorAccessor 3rd arg

**None — the subject never fired.** No accessor in any of the four kernels passes a third argument;
every construction is 2-arg. Nothing to drop.

> Near-miss to not misread: `sparsity_pagesize` is a **read size** handed to `noc.async_read`, not
> an accessor constructor argument. It stays as a named CTA — the forks read
> `get_arg(args::sparsity_pagesize)` under `#ifdef SPARSITY`.

### CB endpoints

Six CBs. Census is per node; the in0 sender covers `{start_core}` and the in0 receiver covers
`all_cores \ {start_core}` — **disjoint**, so their shared `in0` producer role is an ordinary
per-node 1:1, *not* a multi-binding.

| CB | Disposition |
|---|---|
| `c_0` in0 | ordinary DFB — in0 sender **or** in0 receiver PRODUCER (disjoint nodes), compute CONSUMER |
| `c_1` in1 | ordinary DFB — in1 sender/writer PRODUCER, compute CONSUMER |
| `c_4` out | ordinary DFB — compute PRODUCER, in1 sender/writer CONSUMER |
| `c_5` interm0 | **self-loop** — compute is the only toucher (`reserve_back`/`push_back` *and* `wait_front`/`pop_front`); bind it PRODUCER **and** CONSUMER |
| `c_6` sparsity | **self-loop** — in0 sender is the only toucher (fork `:202`, `:483`–`:485`) |
| `c_7` in1 sparsity / indices | **self-loop** — in1 sender/writer is the only toucher (fork `:283`, `:331`, `:339`) |

**No multi-binding flag anywhere. No dead CB to drop.**

**Conditional alias group — `{out, interm0}`.** Legacy puts `c_4` and `c_5` on **one**
`CBDescriptor` with two `format_descriptors` when `interm0_data_format == output_data_format`
(`:712`–`:729`), and on two independent descriptors otherwise (`:681`–`:711`). So:

- **formats equal** → two `DataflowBufferSpec`s naming each other in
  `advanced_options.alias_with`, **both sized from `out_CB_size`** (the alias legality rule wants
  identical `num_entries * entry_size`).
- **formats differ** → no aliasing; `interm0` sized from `interm0_CB_size`.

This is [Aliased DFBs], gated per config — *not* same-FIFO aliasing (the two indices have distinct
page sizes in general and independent FIFO pointers).

Three named CB args point at indices this factory never allocates — `cb_in0_sharded` → `c_2`
(`:500`), `cb_bias` → `c_3` (`:527`), `cb_in0_transposed` → `c_10` (`:589`). Each is read by its
kernel only under a feature this factory never enables, so **declare no binding for any of them**;
they are dead arguments, not buffers. (Audit *Misc anomalies* records why they exist.)

## Watch for

- **Shared kernels — you are on rung 1 (reuse) for all four, and the forks are read-only.** Every
  kernel this factory binds is *lent*: it lives in matmul's own `device/kernels/` tree and the dense
  `MatmulDeviceOperation` factories bind it too. A `_metal2` fork already sits beside each original
  **and already has ported consumers**, so per the shared-kernel Caution you **point
  `KernelSpec::source` at the fork and adopt its vocabulary**. Do not create a second fork, do not
  copy one into the sparse tree, and **do not edit one** — if a fork genuinely does not fit,
  re-derive the need from the legacy factory first, then stop and hand off.

  | Bind this | Instead of |
  |---|---|
  | `…/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp` | `…_in0_sender_padding.cpp` (`:494`) |
  | `…/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp` | `…_in0_receiver.cpp` (`:509`) |
  | `…/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` | `…_in1_sender_writer_padding.cpp` (`:521`) |
  | `…/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp` | `…_activation.cpp` (`:579`) |

  Remaining unmigrated consumers of the **legacy** copies (the sunset list, **not** authorization to
  convert in place): `matmul_multicore_reuse_mcast_1d_program_factory.cpp` and
  `matmul_multicore_reuse_mcast_2d_program_factory.cpp` descriptor paths.

- **The forks already carry the sparsity region, define-gated, and nothing has ever set the
  define.** Each fork's header note names `SPARSITY` among its optional resource families, and the
  two readers already read `batchB`, `bcast_A`, `get_batch_from_reader`, `num_active`,
  `sparsity_pagesize`, `num_batch_compute`, `compact_output`. **No factory in the tree sets
  `SPARSITY` today** — this port is the first, so that region has never been exercised through a
  Metal 2.0 spec even though the code is checked in. Expect the first failures (if any) to be there,
  not in the shared mcast skeleton.

  You must emit `SPARSITY` to **both** the in0 sender and the in1 sender/writer. It is
  unconditional for this factory: `sparsity` is a required input (`input_tensors.at(2)`) and
  `batchB = get_batch_size(bshape) >= 1`, so the legacy `if constexpr (batchB > 0)` guard is always
  taken.

- **⚠ The DM processor assignment is the *opposite* of the dense factory's — carry the sparse
  values.** This is exactly the silent perf-regression the recipe's hw_config section warns about:
  no test distinguishes the two, and the dense metal2 factory sitting next to you has them the other
  way round.

  | Kernel | Sparse legacy (`:505`, `:516`, `:533`) | Dense metal2, for contrast |
  |---|---|---|
  | in0 sender | `RISCV_0`, `in0_noc` (= `NOC_1`) | `RISCV_1`, `in0_noc` |
  | in0 receiver | `RISCV_0`, `in0_noc` | `RISCV_1`, `in0_noc` |
  | in1 sender/writer | `RISCV_1`, `in1_noc` (= `NOC_0`) | `RISCV_0`, `in1_noc` |

  Both sparse triples happen to coincide with a *default* today — in0 with the **writer** default
  and in1 with the **reader** default — because `preferred_noc_for_dram_write` is `NOC_1` and
  `..._read` is `NOC_0` on every arch. Use your judgement on helper-vs-explicit; whichever you pick,
  the resolved `(processor, noc, noc_mode)` must match the table above, and `noc_mode` stays default
  `DM_DEDICATED_NOC` on both.

- **`opt_level`: the compute `KernelSpec` needs an explicit `O3`.** `grep -n opt_level` the legacy
  factory returns **nothing** — and for a `ComputeConfigDescriptor` an absent field still resolves
  to `O3`, while Metal 2.0's `CompilerOptions` defaults to `O2`. The two DM specs need nothing
  (legacy `O2` → Metal 2.0 `O2`).

- **Compute `hw_config` — Style A, and nothing is dropped.** The factory resolves a TTNN
  `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:138`) and passes **all four** knobs
  to its `ComputeConfigDescriptor` (`:593`–`:598`), including `dst_full_sync_en`. So
  `to_compute_hardware_config(device->arch(), config)` is faithful with **no** pinned field —
  unlike the dense metal2 factory, which has to force `double_buffer_dest(compute_hw) = true`
  because it drops the knob. Do not copy that pin. (`packer_l1_acc` is resolved and has no Metal 2.0
  counterpart; it feeds the `PACKER_L1_ACC` define, which is correct and unchanged.)

- **`unpack_modes` — reindex, translate, and add the entries Metal 2.0 newly requires.** Legacy
  (`:568`–`:572`) builds a `NUM_CIRCULAR_BUFFERS`-long vector of `Default` and sets
  `[c_5] = UnpackToDestFp32` when `fp32_dest_acc_en && interm0_data_format == Float32`. Under Metal
  2.0 that becomes a `Table` keyed by DFB name, and the validator *additionally* requires an
  explicit entry for **every** Float32 DFB the compute kernel consumes when `enable_32_bit_dest` is
  on. Under `fp32_dest_acc_en`, add one entry per Float32-format DFB the compute kernel consumes
  (`in0`, `in1`, `interm0`) — `UnpackToDest` for `interm0` when the legacy condition holds,
  `UnpackToSrc` for the rest. `out` is produced, not consumed, so it takes none.

- **`FUSE_ACTIVATION = "0"` is a dead define** (`:454`). No matmul compute kernel, legacy or fork,
  reads the name. Carrying it is inert; the audit records it under *Misc anomalies*.

- **RTA varargs:** none. Every runtime arg is a distinct field read once — name all of them. The
  per-core loop is node-first; `AddRuntimeArgsForNode` transposes it without restructuring the loop.

- **The in1 sender/writer's `num_blocks_w_dim` RTA has a per-core last-column value.** Legacy pushes
  a trailing positional arg (`:858`–`:862`) that is `last_out_num_blocks_w` on the last column and
  `out_num_blocks_x` elsewhere. The fork reads it as `args::last_num_blocks_w_dim`, under
  `#ifndef OUT_SHARDED` — which this factory never defines, so it is always live. Do not let it ride
  the tail of the positional block unnamed.

- **`batchA` is passed twice to the in0 sender**, as both `in0_B` and `in1_B` (`:360`, `:361`). That
  is correct for this op; the duplicated literal just reads like a copy-paste bug in the positional
  list.
