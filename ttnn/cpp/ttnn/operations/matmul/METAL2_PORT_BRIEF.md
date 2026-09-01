# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Port scope: ONE factory — `MatmulMultiCoreReuseMcast2DProgramFactory`.** The op has eight factories
across two DeviceOperations; only this one was audited and only this one is cleared. Do not widen.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A — no site)

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers` *(carry this line into the port report's Provenance section)*

> ## ⚠ Read this before you start — a precondition, not a port-time decision
>
> **The method you must reshape is called from outside this op, and the obvious workaround is
> closed.** This factory targets `CustomProgramSpecFactoryConcept`, which requires
> `override_runtime_arguments` to return `ProgramRunArgs`. Two CCL device operations call the current
> void version directly:
> - `experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:241`
> - `experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp:147`
>
> You cannot add a second overload: the concept keys on `decltype(&T::override_runtime_arguments)`,
> which requires a **single unambiguous** overload, so a second one breaks the detection outright.
> You cannot change it in place either — that breaks both CCL ops, which are separate device
> operations with their own readiness rows and are not porting with you. And editing their callers is
> outside the port's scope boundary.
>
> **The audit raised this with the invoker as an open question.** If you were handed this brief
> without an answer, stop and ask before converting anything — this is the most likely reason the
> port would capitulate, and it is cheaper to resolve now than mid-conversion.

## TTNN factory analysis

- **Current concept:** `descriptor` — `create_descriptor` returning a `ProgramDescriptor`
  (`…mcast_2d_program_factory.hpp:41`).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** — the factory has an
  `override_runtime_arguments` (`…mcast_2d_program_factory.cpp:3382`), so the op owns the entire
  cache-hit refresh. **Translate that method; do not delete it.** Deleting drops the factory to the
  base concept, where the framework patches tensor bindings and nothing else — silently discarding
  any non-tensor refresh. A half-finished translation and a deliberate one are indistinguishable to
  the framework.
- **Custom `compute_program_hash`:** none framework-visible — the op uses the default reflection
  hash. You will find a `compute_descriptor_program_hash` helper at
  `device/matmul_device_operation.hpp:50` that is *deliberately* not named `compute_program_hash`,
  plus a pybind exposing it under that name. **Leave all of it alone.**
- **Gate-cleared, confirmed absent:** a non-`none` `TensorParameter relaxation` ·
  `get_dynamic_runtime_args` · smuggled pointers (verified — every address is rebound). A pybound
  `create_descriptor` **is** present; removing it is port work.

## Construct — to do

### Translating the override — this is the centre of the port

Read the cache-hit contract in `ttnn_factory.md` first. On this concept the framework refreshes
**nothing** on your behalf — if your returned `ProgramRunArgs` omits a tensor binding, it stays
frozen at the cache-miss address for the life of the cache entry, and the failure appears only on
cache hits, only once incoming tensors stop landing at the first call's addresses.

The good news: `override_runtime_arguments_impl` (`:3056-3139`) is short and **every statement it
makes is an address refresh**. It touches no non-address runtime argument at all.

| Ported-from statement | Condition | Goes to |
|---|---|---|
| `UpdateDynamicCircularBufferAddress(program, cb_src2, in0)` (`:3101`) | `src0_sharded` | `TensorArgument` for **in0** |
| `reader_runtime_args[0] = in0.address()` (`:3106`) | `!src0_sharded` | `TensorArgument` for **in0** |
| `writer_runtime_args[0] = in1.address()` (`:3114`) | always | `TensorArgument` for **in1** |
| `writer_runtime_args[7] = out.address()` (`:3115`) | always | `TensorArgument` for **output** |
| `writer_runtime_args[18] = bias_mesh->address()` (`:3117`) | bias present | `TensorArgument` for **bias** |
| `writer_runtime_args[2] = out.address()` (`:3125`) | always | `TensorArgument` for **output** |
| same, other-NOC receiver group (`:3132`) | receivers distinct | `TensorArgument` for **output** |
| `UpdateDynamicCircularBufferAddress(program, cb_output, out)` (`:3137`) | `out_sharded` | `TensorArgument` for **output** |

**Eight statements collapse to four `TensorArgument`s** — in0, in1, output, bias — and
`kernel_run_args` comes out **empty**. The sharded/interleaved branches collapse too: a borrowed DFB
draws its backing L1 address from the corresponding `tensor_args` entry automatically, so supplying
the tensor covers both the CB-address form and the RTA form.

Every io tensor is refreshed, so there is **no omission to justify** — check the set both ways and
you should find it complete.

**Addresses become bindings, never runtime-arg values.** Re-expressing any of the rows above as a
runtime arg is the smuggling anti-pattern the binding model exists to prevent.

### Tensor bindings — four, all Case 1

`in0`, `in1`, `output`, and (conditionally) `bias`. Each already arrives as a `MeshTensor` reference
via the variant mechanism and is consumed kernel-side through a `TensorAccessor`. Straight
translation to `TensorParameter` / `TensorBinding`; the address slots and their `TensorAccessorArgs`
plumbing both disappear, so re-index what remains. No Case 2 site — the `get_bank_base_address`
bridge is not needed anywhere.

### CB endpoints

Four semaphores (`SemaphoreSpec` × 4, straightforward). Three CB dispositions need attention:

- **Four borrowed-memory CBs**, each conditional on the matching operand being sharded:
  `c_0 ← in0_tensor` (`:983`), `c_1 ← in1_tensor` (`:999`), `c_2 ← in0_tensor` (`:1015`),
  `c_4 ← out_tensor` (`:1045`, `:1093`). Each becomes `DataflowBufferSpec::borrowed_from` naming the
  matching `TensorParameter`. `c_2` (`cb_src2`) and `c_4` (`cb_output`) are exactly the two the
  override patches — their bindings are what your translated `tensor_args` must cover.
- **An aliased-DFB group of two *or three* members, config-dependent.** When output and intermediate
  share a buffer, one `CBDescriptor` carries `c_4` **and** `c_5` (`:1075-1093`); when the bias reload
  alias is active it carries a third index, `cb_intermed0_alias` (`:1062`, `:1087`). Port as two or
  three `DataflowBufferSpec`s whose `advanced_options.alias_with` forms a **strict clique** (every
  member naming every other), same total size, same bound kernels. In the non-shared branch
  (`:1040-1067`) they are separate descriptors and no aliasing applies. **Derive the group size per
  instantiation** — do not assume three.
- **`c_6` is a 32-byte scratch CB** (`:1025`) with no tensor backing. Run the toucher census on it
  specifically: one toucher → **self-loop** (bind that kernel PRODUCER + CONSUMER); zero → **dead-CB
  drop** (record with `file:line`).

### `unpack_to_dest_mode` → `unpack_modes`

The factory builds an `unpack_to_dest_mode` vector and marks one CB `UnpackToDestFp32` (`:948-953`),
then sets it on the `ComputeConfigDescriptor` (`:959`). All three hazards are live:

1. **Reindex** — legacy `vector<UnpackToDestMode>` by CB id → `Table<DFBSpecName, UnpackMode>` by
   name. Trace what the computed vector resolves to per CB; it is not a literal.
2. **Translate values without inverting them** — `UnpackToDestFp32` → `UnpackMode::UnpackToDest`;
   `Default` → `UnpackToSrc`, normally expressed by omitting the entry. Reversing this flips the
   precision/perf tradeoff with no compile or test signal.
3. **The newly-required explicit entry** — any Float32 DFB the compute kernel consumes with
   `enable_32_bit_dest` on needs one, where legacy defaulted silently. The trigger is the **DFB's
   format**, not the op's tensor dtypes.

A conditionally-bound DFB's entry must be gated on the same condition as its binding — the validator
rejects a key naming a DFB the kernel doesn't bind.

### `opt_level` — set `O3` explicitly on every compute KernelSpec

`grep -n opt_level` returns nothing. An absent `KernelDescriptor::opt_level` still resolves to the
legacy per-kernel-type default — **`O3` for a `ComputeConfigDescriptor`** — while Metal 2.0's
`CompilerOptions` defaults to `O2` for both kinds. Silent perf loss with no test net. DM kernels need
nothing.

### Hardware config — Style A, and nothing dropped

The factory resolves a TTNN `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:3464-3465`),
and the descriptor builder both **receives** `dst_full_sync_en` (signature `:44`, passed at `:3492`)
and **sets** it (`:958`). All four helper-covered knobs are set, so
`to_compute_hardware_config(device->arch(), config)` translates faithfully with nothing to reapply by
hand. `packer_l1_acc` has no Metal 2.0 counterpart; no action.

**DM configs are custom, including split-NOC variants — replicate, don't use the helpers.** The
builder emits `RISCV_1 / in0_noc`, `RISCV_0 / in1_noc`, and split variants
`RISCV_0 / in1_split_noc` (`:847`) and `RISCV_1 / in0_split_noc` (`:860`). Copy every field verbatim
into a `DataMovementGen1Config`. `create_reader_datamovement_config` /
`create_writer_datamovement_config` would substitute the default triple and regress silently.

### Device-op-class edits this port forces

1. **Delete the pybound factory entry point.** `matmul_nanobind.cpp:1293` opens an
   `nb::class_<…Mcast2DProgramFactory>` block whose only member is `create_descriptor`. Leave the
   separate `nb::class_<MatmulDeviceOperation>` block at lines 1222-1237 untouched.
2. **Drop the pybind-hook-only parameter** — `create_descriptor`'s `core_range_set`, which exists
   only for that hook.

Exception 3 does not apply — the op has a proper `program_factory_t` variant.

## Watch for

- **You are working in a file with two builders; only one is yours.**
  - **`create_program_mcast_in0_in1_descriptor` (39-1568)** — reached from `create_descriptor`
    (`:3392`). **This is your scope.**
  - **`create_program_mcast_in0_in1` (1571-3054)** — the legacy `Program` builder, reached only from
    `matmul_multi_core_reuse_mcast_2d_optimized_helper` (`:3532`), which only the two CCL ops call.
    **Not yours.**

  The two use *opposite* conventions: yours rebinds every address to a tensor reference; the legacy
  one writes raw addresses (`:2759`, `:2807`, `:2820`, `:2852`, `:2952`) and relies on the shared
  override. If a search drops you at line 2800 and you convert what you find, you will have modified
  the CCL ops' builder.

- **All six kernels are shared, and none has a `_metal2` fork — you create six.**

  | Kernel | Also bound by | Rung |
  |---|---|---|
  | `in0_sender_padding` | mcast_1d file (2 factories), sparse device-op, sparse factory | **2 — create** |
  | `in0_sender_receiver_padding_block_sharded` | mcast_1d file (2 factories) | **2 — create** |
  | `in0_receiver` | mcast_1d file (2 factories), sparse factory | **2 — create** |
  | `in1_sender_writer_padding` | mcast_1d file (2 factories), sparse factory | **2 — create** |
  | `in1_receiver_writer_padding` | mcast_1d file (2 factories) | **2 — create** |
  | `bmm_large_block_zm_fused_bias_activation` | BatchedHS, Optimized, McastDRAMSharded, sparse, mcast_1d file | **2 — create** |

  The rung-1 check was run locationally: `find` over `matmul/device/kernels/` returns **zero**
  `*_metal2*` files. **Name the forks' bindings for the kernel, not for this factory** — those names
  become the interface every later consumer inherits, and they will not be able to rename them.
  Note `matmul_multicore_reuse_mcast_1d_program_factory.cpp` hosts **two** factories, so each row
  above understates the remaining-consumer count by one.

- **`transpose_mcast` is a configuration axis, not a flag.** It swaps the row/column roles of the 2D
  grid, changing which cores are senders and which are receivers — and therefore which kernel binds
  which DFB in which role. Map every DFB's producer/consumer **per `transpose_mcast` value**; a role
  assignment derived from one setting will mis-bind the other.

- **Two receiver kernel groups, distinguished only by NOC setup.** Both bind
  `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` (`:802`, `:834`) with different
  `DataMovementConfigDescriptor`s, and the override refreshes them separately behind a
  distinct-ids guard (`:3127`). That is two `KernelSpec`s of one source over disjoint node sets —
  **preserve the multiplicity**; do not merge them into one.

- **RTA varargs: none — name every argument.** The kernels read args through a running
  `rt_args_idx++` counter, but every read is a **distinct field taken once** in a block at the top.
  A running counter is not a vararg signal. No loop-indexed read, no data-selected index, no
  sentinel scan — nothing here justifies `get_vararg`.

- **Dead placeholder addresses will mislead a text search.** `:1248`, `:1300`, `:1313`, `:1457`
  compute a `.address()` that the variant rebinding overwrites tens to hundreds of lines later
  (`:1273`, `:1439-1442`, `:1530`). They are not smuggled pointers. Don't "fix" them and don't be
  alarmed by them.

- **Locate the tests and confirm the set with your invoker.** This factory serves block-sharded 2D
  matmul and is flagged `ProgramFactory used in llama? = yes` on the readiness sheet, so its coverage
  likely spans model-level tests as well as op unit tests. Search broadly rather than assuming a
  mirror path.
