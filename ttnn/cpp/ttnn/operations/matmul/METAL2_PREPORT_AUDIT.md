# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMultiCoreReuseMcast2DProgramFactory`.**
The op directory holds two DeviceOperations and eight ProgramFactories; this audit covers a single
factory by request. The other seven are named below for disambiguation only and were **not**
audited — no statement in this report is a verdict on any of them.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMultiCoreReuseMcast2DProgramFactory`** ← **audited**
    (`device/factory/matmul_multicore_reuse_mcast_2d_program_factory.{hpp,cpp}`)
  - `MatmulMultiCoreProgramFactory` — not audited
  - `MatmulMultiCoreReuseOptimizedProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` — not audited
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

> **Scoping note — the file contains a second builder this factory never reaches.**
> `matmul_multicore_reuse_mcast_2d_program_factory.cpp` (3,552 lines) holds two program builders:
> - **`create_program_mcast_in0_in1_descriptor`** (39-1568) — reached from
>   `MatmulMultiCoreReuseMcast2DProgramFactory::create_descriptor` (3392). **This is this factory's
>   dispatch path** and the scope of every finding below.
> - **`create_program_mcast_in0_in1`** (1571-3054) — the legacy `Program`-based builder, reached only
>   from `matmul_multi_core_reuse_mcast_2d_optimized_helper` (3532), which is called exclusively by
>   two **external CCL device operations**: `all_gather_matmul_async` (`…_program_factory.cpp:82`)
>   and `matmul_reduce_scatter_async` (`…_program_factory.cpp:120`). Those are separate ops with
>   their own readiness rows and are out of scope here.
>
> The distinction is load-bearing: the legacy builder writes raw buffer addresses into runtime args
> **without** rebinding them (`:2759`, `:2807`, `:2820`, `:2852`, `:2952`), while the descriptor
> builder rebinds every one. A reader scanning this file will find un-rebound addresses that belong
> to a different consumer entirely.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **GREEN** (for `MatmulMultiCoreReuseMcast2DProgramFactory` only) |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMultiCoreReuseMcast2DProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 6 in-scope kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | **Ok** — all six kernels are matmul-owned; no donor function-call escapes |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A · N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** (framework-visible) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** — `:3382`, returns **void**, delegating to `override_runtime_arguments_impl` (`:3056`) |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — `matmul_nanobind.cpp:1293` |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** |
| *Port work* — Offset base pointer | **none** — no host-folded offset anywhere in the file |
| *Port work* — Tensor bindings (per binding) | 4 bindings, all **Case 1**, all already rebound to tensor references |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — every construction is 2-arg |
| *Port work* — CB endpoints | aliased-DFB group, four borrowed-memory CBs, one scratch CB |

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, alongside this file).

Every gate cleared. Two things make this port materially different from a default one, and both are
structural rather than incidental:

1. **It targets `CustomProgramSpecFactoryConcept`**, so the op's `override_runtime_arguments` owns
   the *entire* cache-hit refresh — including tensor bindings, which the framework does **not**
   supply on this concept. The good news is that the ported-from override is small and fully
   inventoried below, and it refreshes nothing but addresses.
2. **The method the port must reshape is called from outside the op.** Two CCL device operations
   invoke `MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments` directly, and the
   concept detection requires a *single* override whose return type is `ProgramRunArgs` — so the
   obvious "add an overload" escape is closed. This needs a coordination decision before the port
   starts; see [Heads-ups](#heads-ups) and the Questions section.

There is no blocked code path within this factory and no subset to carve.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreReuseMcast2DProgramFactory`) reads **`yes`**,
  with `Diego validation = yes`, `Op Classification = PD Op (custom)`, `Execution Model = SPMD`,
  `Porting Target = CustomProgramSpecFactoryConcept`. Lightweight cross-check clean on every primary
  column:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returning `ProgramDescriptor` (hpp:41) | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op — see below | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across the op's `.cpp` / `.hpp` | ✓ |
  | `Override runtime args method?` | `yes` | Present at `:3382` (hpp:34) | ✓ |
  | `Pybind descriptor` | `PR` | Present at `matmul_nanobind.cpp:1293` | ✓ (in-flight PR; still on this checkout) |
  | `Smuggled pointer` | `no` | **Independently verified** — every address on the descriptor path is rebound to a `MeshTensor` reference before dispatch (below) | ✓ |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1 — no phantom or missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` row, and
  `Op-owned tensors?` is empty.

  **`Smuggled pointer = no` verified against the code.** The descriptor builder assembles runtime
  args as a plain `std::vector<uint32_t>` containing *placeholder* `.address()` values, then converts
  to a `std::vector<std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>>` and overwrites
  every address slot with the tensor reference before `emplace_runtime_args`:

  | Emplace site | Rebinds | Placeholder it overwrites |
  |---|---|---|
  | `:1274` (in0 sender, interleaved) | `in0_args[0] = in0_tensor` (`:1273`) | `:1248` |
  | `:1444` (in1 sender/writer) | `[0] = in1_tensor`, `[7] = out_tensor`, `[18] = *bias_mesh` (`:1439-1442`) | `:1300`, `:1313`, and the bias site |
  | `:1533`, `:1537` (in1 receiver/writer, both NOC setups) | `in1_recv_variant[2] = out_tensor` (`:1530`) | `:1457` |

  The three emplace sites that skip the variant form carry no address at all: `:1241` / `:1243` are
  the in0-**sharded** branch, whose vector (`:1220-1238`) holds only core and mcast NOC coordinates
  (in0 arrives through the borrowed `cb_src2`), and `:1286` / `:1290` are the in0 receivers, whose
  args are just the mcast sender's NOC coordinates.

  **On the custom hash.** The device-op declares `compute_descriptor_program_hash`
  (`device/matmul_device_operation.hpp:50`) with a comment that it is *deliberately* not named
  `compute_program_hash`, so the framework does not detect a custom cache hash; it is reached only
  through a pybind alias. The framework uses the **default reflection hash**. Nothing for the port
  to touch.

- **Device 2.0 (every kernel used): GREEN.** All six kernels reachable from this factory are
  structurally Device 2.0 — `Noc` from `noc.h`, `DataflowBuffer` wrappers, `TensorAccessor`. Zero
  hits for broad Device-1.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`,
  raw `noc_async_read(` / `noc_async_write(`, raw `noc_semaphore_*`) and no non-sanctioned CB-index
  free-function holdovers. All six are matmul-owned, so there is no donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Bound at |
  |---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | `:765` |
  | `dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | `:725`, `:745` |
  | `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | `:819`, `:851` |
  | `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `:783` |
  | `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | `:802`, `:834` |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | `:904` |

  Several call `get_local_cb_interface(...)` and/or `get_tile_size(...)`. Both are on the recipe's
  **sanctioned** list, which "does not turn on what object is in scope," so they stay sanctioned even
  where a `DataflowBuffer` is in scope. Not Device 2.0 violations; they are Metal 2.0 port-stage
  rewrites (onto the object, or kept in free-function form with the binding token where the value is
  `constexpr`).

- **Feature compatibility: GREEN.** Every Appendix A entry scanned against the factory and its six
  kernels. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_index` / `remote_cb_config`, no `global_cb` parameter anywhere in the file |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `cb_descriptor_from_sharded_tensor`. The two `UpdateDynamicCircularBufferAddress` calls in the override (`:3101`, `:3137`) are the **three-argument** no-offset form, which Appendix A's false-positive guard excludes |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`. The factory declares four ordinary `SemaphoreDescriptor`s |

- **Offset base pointers: GREEN.** A scan for `address() +`, `addr + …` and `+ …offset…` across the
  whole file returns **zero** hits, and on the descriptor path every `.address()` is a placeholder
  overwritten by a tensor reference. No Type 1, no Type 2, no Type 3, no Type 4. The checked-in
  offset triage lists no matmul row, and my own scan agrees.

- **TensorAccessor 3rd argument: N/A.** No accessor in any of the six kernels passes a third
  (page-size) argument — every construction is 2-arg. The subject never fires.

---

## Port-work summary  *(mirrors the brief)*

- **Target concept: `CustomProgramSpecFactoryConcept`.** The override is what selects it, and the
  port *translates* rather than deletes it — deleting drops the factory to the base concept, where
  the framework patches tensor bindings and nothing else, silently discarding any non-tensor refresh.

- **The ported-from override, fully inventoried.** `override_runtime_arguments_impl` (`:3056-3139`)
  is short, and **every statement it makes is an address refresh** — it touches no non-address
  runtime argument at all. That makes the translation unusually clean:

  | Ported-from statement | Condition | Metal 2.0 destination |
  |---|---|---|
  | `UpdateDynamicCircularBufferAddress(program, cb_src2, in0)` (`:3101`) | `src0_sharded` | `TensorArgument` for **in0** |
  | `reader_runtime_args[0] = in0.address()` per in0-sender core (`:3106`) | `!src0_sharded` | `TensorArgument` for **in0** |
  | `writer_runtime_args[0] = in1.address()` per in1-sender core (`:3114`) | always | `TensorArgument` for **in1** |
  | `writer_runtime_args[7] = out.address()` per in1-sender core (`:3115`) | always | `TensorArgument` for **output** |
  | `writer_runtime_args[18] = bias_mesh->address()` per in1-sender core (`:3117`) | bias present | `TensorArgument` for **bias** |
  | `writer_runtime_args[2] = out.address()` per in1-receiver core (`:3125`) | always | `TensorArgument` for **output** |
  | same, for the other-NOC receiver group (`:3132`) | two distinct receiver kernels | `TensorArgument` for **output** |
  | `UpdateDynamicCircularBufferAddress(program, cb_output, out)` (`:3137`) | `out_sharded` | `TensorArgument` for **output** |

  **Eight statements collapse to four `TensorArgument`s** — in0, in1, output, bias — and
  `kernel_run_args` comes out **empty**. The sharded / interleaved branches collapse too: a borrowed
  DFB draws its backing L1 address from the corresponding `tensor_args` entry automatically, so
  supplying the tensor covers both the CB-address and the RTA form. Every io tensor is refreshed, so
  there is **no omission to justify** — the clean case the recipe describes.

  Addresses become **bindings, not runtime-arg values**. Re-expressing any of them as a runtime arg
  is the smuggling anti-pattern the binding model exists to prevent.

- **Tensor bindings — four, all Case 1.** `in0`, `in1`, `output`, and (conditionally) `bias`. Each is
  already delivered as a `MeshTensor` reference through the variant mechanism and consumed kernel-side
  through a `TensorAccessor`. Straight translation to `TensorParameter` / `TensorBinding`; the address
  slots and their `TensorAccessorArgs` plumbing both disappear. No Case 2 site, so the
  `get_bank_base_address` bridge is not needed anywhere.

- **CB endpoints.** Four semaphores (`SemaphoreSpec` × 4, straightforward) and a CB set with three
  dispositions worth attention:

  - **Four borrowed-memory CBs**, each conditional on the corresponding operand being sharded:
    `c_0 ← in0_tensor` (`:983`), `c_1 ← in1_tensor` (`:999`), `c_2 ← in0_tensor` (`:1015`),
    `c_4 ← out_tensor` (`:1045`, `:1093`). Each becomes `DataflowBufferSpec::borrowed_from` naming
    the matching `TensorParameter`. Note `cb_src2` (`c_2`) and `cb_output` (`c_4`) are exactly the
    two the override patches — their bindings are what the translated `tensor_args` must cover.
  - **An aliased-DFB group of two or three members, config-dependent.** When output and intermediate
    share a buffer, one `CBDescriptor` carries `c_4` **and** `c_5` (`:1075-1093`); when the bias
    reload alias is active it carries a third index, `cb_intermed0_alias` (`:1062`, `:1087`). Port as
    two or three `DataflowBufferSpec`s whose `advanced_options.alias_with` forms a **strict clique**,
    all the same total size and bound to the same kernels. In the non-shared branch (`:1040-1067`)
    they are separate descriptors and no aliasing applies. **Derive the group size per instantiation.**
  - **`c_6` is a 32-byte scratch CB** (`:1025`), no tensor backing. Run the toucher census on it
    specifically: one toucher → **self-loop**; zero → **dead-CB drop**.

- **`unpack_to_dest_mode` → `unpack_modes`, with all three hazards live.** The factory builds an
  `unpack_to_dest_mode` vector and marks one CB `UnpackToDestFp32` (`:948-953`), then sets it on the
  `ComputeConfigDescriptor` (`:959`). The translation must **reindex** (legacy `vector` by CB id →
  `Table<DFBSpecName, UnpackMode>` by name), **translate values without inverting them**
  (`UnpackToDestFp32` → `UnpackMode::UnpackToDest`; `Default` → `UnpackToSrc`, normally expressed by
  omission), and satisfy the **newly-required explicit entry** for any Float32 DFB the compute kernel
  consumes with `enable_32_bit_dest` on. A conditionally-bound DFB's entry must be gated on the same
  condition as its binding.

- **Hardware config — Style A, and no dropped field.** The factory resolves a TTNN
  `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:3464-3465`), and — unlike some
  siblings — the descriptor builder **does** receive `dst_full_sync_en` (signature `:44`, passed at
  `:3492`) and **does** set it on the `ComputeConfigDescriptor` (`:958`). All four helper-covered
  knobs are set, so `to_compute_hardware_config(device->arch(), config)` translates faithfully with
  nothing to reapply by hand. `packer_l1_acc` has no Metal 2.0 counterpart; no action.

- **DM configs are custom, including split-NOC variants — replicate, do not use the helpers.** The
  builder emits explicit `DataMovementConfigDescriptor`s with `RISCV_1 / in0_noc`,
  `RISCV_0 / in1_noc`, and **split** variants `RISCV_0 / in1_split_noc` (`:847`) and
  `RISCV_1 / in0_split_noc` (`:860`). Copy every field verbatim into a `DataMovementGen1Config`;
  reaching for `create_reader_datamovement_config` / `create_writer_datamovement_config` would
  substitute the default triple and regress silently.

- **`opt_level` — absent.** `grep -n opt_level` returns nothing. An unset
  `KernelDescriptor::opt_level` still resolves to the legacy per-kernel-type default — **`O3` for a
  `ComputeConfigDescriptor`**, `O2` for DM — while Metal 2.0's `CompilerOptions` defaults to `O2` for
  both. Set `O3` explicitly on every compute `KernelSpec` the port builds.

- **RTA varargs: none.** The kernels read arguments through a running `rt_args_idx++` counter, but
  every read is a **distinct field taken once** in a block at the top — no loop-indexed read, no
  data-selected index, no sentinel scan. A running counter is not a vararg signal. All become named
  RTAs.

- **Device-operation-class edits the port forces** — two sanctioned exceptions:
  1. **Remove the pybound factory entry point.** `matmul_nanobind.cpp:1293` opens an
     `nb::class_<ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory>` block whose only member is
     `create_descriptor`. Deletion is mandatory once that method goes; user-visible API change.
  2. **Drop the pybind-hook-only parameter.** `create_descriptor`'s fourth argument,
     `const std::optional<CoreRangeSet>& core_range_set`, exists only for that hook. Drop it.

  Exception 3 does not apply — the op has a proper `program_factory_t` variant.

---

## Heads-ups  *(mirrors the brief)*

- **⚠ The override the port must reshape is called from outside this op, and the obvious workaround
  is closed.** Two CCL device operations invoke it directly:
  - `experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:241`
  - `experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp:147`

  Both also build through `matmul_multi_core_reuse_mcast_2d_optimized_helper` (`:82` and `:120`
  respectively), which returns a `CachedProgram` carrying `shared_variables_t` — exactly what the
  current void override consumes.

  `CustomProgramSpecFactoryConcept` requires the override to return `ProgramRunArgs`, and the
  detection is keyed on `decltype(&T::override_runtime_arguments)`, which requires a **single,
  unambiguous** overload. So adding a second `ProgramRunArgs`-returning overload alongside the void
  one would make that expression ambiguous and break the concept detection — the natural escape does
  not work. Changing the existing method in place breaks both CCL ops, which are separate device
  operations with their own readiness rows and are not porting in this change.

  This is a host-side analogue of the shared-kernel problem, and the recipe has no named pattern for
  it. **It needs a decision before the port starts** — see Questions. It is not an audit gate (every
  gate cleared), but it is the most likely reason this port would capitulate.

- **All six kernels are shared, and none has a `_metal2` fork.** Every kernel this factory binds is
  also bound elsewhere in the matmul directory, so none can be converted in place.

  | Kernel | Also bound by | Fork? | Rung |
  |---|---|---|---|
  | `in0_sender_padding` | mcast_1d file (2 factories), sparse device-op, sparse factory | no | **2 — create** |
  | `in0_sender_receiver_padding_block_sharded` | mcast_1d file (2 factories) | no | **2 — create** |
  | `in0_receiver` | mcast_1d file (2 factories), sparse factory | no | **2 — create** |
  | `in1_sender_writer_padding` | mcast_1d file (2 factories), sparse factory | no | **2 — create** |
  | `in1_receiver_writer_padding` | mcast_1d file (2 factories) | no | **2 — create** |
  | `bmm_large_block_zm_fused_bias_activation` | BatchedHS, Optimized, McastDRAMSharded, sparse, mcast_1d file | no | **2 — create** |

  The rung-1 check was run **locationally**: `find` over
  `matmul/device/kernels/` returns **zero** `*_metal2*` files. So a port of this factory creates
  **six forks**, and whatever binding vocabulary the first one uses becomes the interface every
  later consumer inherits — worth agreeing across the matmul factories before any of them ports.

  Note that `matmul_multicore_reuse_mcast_1d_program_factory.cpp` appears once per row but hosts
  **two** factories, so each of those rows represents two remaining consumers, not one.

- **`transpose_mcast` is a real configuration axis, not a flag.** It swaps the row/column roles of
  the 2D grid, which changes which cores are senders vs receivers and therefore which kernel binds
  which DFB in which role. Map each DFB's producer and consumer **per `transpose_mcast` value**; a
  role assignment derived from one setting will mis-bind the other.

- **Two receiver kernel groups exist, distinguished only by NOC setup.** The override refreshes
  `mm_kernel_in1_receiver_writer_id` and `mm_kernel_in1_receiver_writer_other_noc_setup_id`
  separately, guarded by a comparison that they are distinct (`:3127`). Both bind
  `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` (`:802`, `:834`) with different
  `DataMovementConfigDescriptor`s. That is two `KernelSpec`s of one source over disjoint node sets —
  preserve the multiplicity; do not merge them.

---

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. No function-call escapes to
  another op's helpers and no file-path escapes — all six kernels are owned and instantiated by
  matmul. Includes are `api/*` (LLK / HAL, donor class 1) plus in-op siblings. **Borrowed kernel
  files: none.**

  Separately, the *host* side does have out-of-directory coupling, in the opposite direction: two CCL
  ops consume this factory's `override_runtime_arguments` and the 2D build helper. That is inventoried
  under Heads-ups because it constrains the port rather than the kernels.

- **Relaxation candidates:** none. No custom hash to mine; the sheet's `TensorParameter relaxation`
  is `none`.

- **TTNN factory analysis:** op-owned tensors — none. MeshWorkload need — none (`Execution Model =
  SPMD`, `Concept = descriptor`). Pybind `create_descriptor` — `matmul_nanobind.cpp:1293`. Other
  risky pybind — the device-op class binding at `matmul_nanobind.cpp:1222-1237`, which survives the
  port untouched. Custom hash — none framework-visible. `get_dynamic_runtime_args` — absent.
  `override_runtime_arguments` — present at `:3382`, returning void; the shape change and its
  external consumers are covered in Port-work and Heads-ups. Target concept —
  `CustomProgramSpecFactoryConcept`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead placeholder `.address()` calls on the descriptor path.** `:1248`, `:1300`, `:1313` and
  `:1457` compute a buffer address that is immediately overwritten by the variant rebinding tens to
  hundreds of lines later. Harmless, but they are what a text search for smuggled pointers finds
  first and cost a reader real time before the rebinding turns up. A comment at the construction site,
  or dropping the placeholder to `0u`, would remove a standing trip hazard. (The sheet correctly
  records `Smuggled pointer = no` for this factory.)

- **The legacy builder in the same file does *not* rebind.** `create_program_mcast_in0_in1`
  (1571-3054) writes raw addresses at `:2759`, `:2807`, `:2820`, `:2852`, `:2952` with no variant
  conversion, relying on the shared override for cache-hit patching. That is correct for its
  consumers (the two CCL ops call the override explicitly), but the two builders in one file having
  opposite conventions is a readability hazard worth a comment.

- **`create_descriptor`'s `core_range_set` parameter is accepted and ignored.** A Python caller
  passing one has it silently discarded. The port removes the parameter, so this resolves itself.

---

## Questions for the user

1. **How should the override's external consumers be handled?** This is the one finding that could
   stop the port. `all_gather_matmul_async` and `matmul_reduce_scatter_async` call
   `MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments` directly, and the concept
   detection forbids adding a second overload. Options as I see them, none of which the porter should
   pick unilaterally: co-port the two CCL ops in the same change; have them stop calling the matmul
   factory's method (moving the logic into their own factories); or defer this factory until they
   port. A decision here is a precondition, not a port-time judgement call.

2. **Fork vocabulary should be agreed before any matmul factory ports.** Six shared kernels have no
   `_metal2` fork yet, and all six are bound by other factories in this directory. Whichever factory
   ports first sets names the others cannot change.

3. **Scope.** Six factories in this directory remain unaudited, plus the sparse device-op's factory.
   Should those follow, and in what order?

## Recipe notes

- **The custom-concept guidance was exactly right about where the risk sits, and this op is a
  reassuring instance of it.** The recipe warns that the silent failure on
  `CustomProgramSpecFactoryConcept` is assuming the framework still patches tensor bindings. Here the
  ported-from override refreshes *only* addresses, so the translated override is almost entirely
  `tensor_args` — which makes the warning land naturally rather than as an abstract caution. Worth
  keeping the emphasis.

- **Gap: no pattern for a factory method consumed outside the op.** The shared-kernel Caution covers
  kernel sources thoroughly, but this factory's `override_runtime_arguments` is a *host* symbol with
  the same problem — two external ops bind it, it cannot be converted in place, and the concept
  detection blocks the add-an-overload workaround. The recipe's scope boundary says "the port does
  not propose changes to files outside the op directory," which correctly forbids editing the CCL
  callers but leaves the porter with no route forward. A short entry — even one that just says "stop
  and surface it" — would save a porter discovering this mid-conversion.

- **Friction: the two-builders-in-one-file shape recurs and costs real scoping effort.** As with
  other matmul factories, deciding which findings are in scope required mapping function boundaries
  and reachability rather than scanning the file. The recipe's guidance to follow kernel references
  rather than directory boundaries has an unstated host-side analogue: follow *dispatch* references,
  not file boundaries.
