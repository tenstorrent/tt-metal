# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMultiCoreReuseMcast1DProgramFactory`.**
The op directory holds two DeviceOperations and eight ProgramFactories; this audit covers a single
factory by request. The other seven are named below for disambiguation only and were **not**
audited — no statement in this report is a verdict on any of them.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMultiCoreReuseMcast1DProgramFactory`** ← **audited**
    (declared `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp:25`)
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — not audited *(shares the same `.cpp`;
    see the scoping note)*
  - `MatmulMultiCoreProgramFactory` — not audited
  - `MatmulMultiCoreReuseOptimizedProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast2DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` — not audited
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

> **Scoping note — two factories share one 5,885-line implementation file, and it matters here.**
> `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` hosts both the audited factory
> and `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`. Their code paths are disjoint:
> - **This factory** builds through `create_descriptor` (5492-5700) into
>   `create_program_mcast_in0_descriptor` (3141-4216) or `create_program_mcast_in1_descriptor`
>   (4217-5150).
> - **The sibling** builds through `matmul_multi_core_reuse_mcast_1d_optimized_` (5152) into the
>   imperative `process_*_program_and_create_override_variables` functions (77-2917).
>
> Every line-referenced finding below was checked against those boundaries and lies in **this**
> factory's reachable code. Where a finding turns on the distinction — and one important one does —
> it is called out explicitly.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **RED** — blocked on one gate |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMultiCoreReuseMcast1DProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 6 in-scope kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | **Ok** — no donor function-call escapes; all six kernels are matmul-owned |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A · N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No** — attributed to `Smuggled pointer = yes` |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** (framework-visible) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** — `…mcast_1d_program_factory.cpp:5482`, returns **void** |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — `matmul_nanobind.cpp:1277-1290` |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` (were the gate to lift) |
| *Port work* — Offset base pointer | **none** — no host-folded offset anywhere |
| *Port work* — Tensor bindings (per binding) | 4 bindings, all **Case 1**, all already delivered as tensor references |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — every construction is 2-arg |
| *Port work* — CB endpoints | assessed — aliased-DFB clique, one scratch CB, otherwise plain 1:1 |

---

## Result

**RED at factory level; no portable subset.** Blocked on a single gate:

- **TTNN factory concept** — the readiness sheet's `Is able to port?` reads **`no`**, attributable to
  `Smuggled pointer (raw buffer addr in RTA/CRTA) = yes`, corroborated by
  `Op Classification = Broken Op` and `Diego validation = no`. Routed to **TTNN**, who fix the op
  before a Metal 2.0 port is possible.

Every other gate cleared: Device 2.0, Feature compatibility (all three Appendix A entries N/A),
Offset base pointers, and TensorAccessor 3rd argument.

**No portable subset exists.** The factory has two code paths — `mcast_in0` and `mcast_in1` — and
they are uniformly in the same state with respect to the blocker (see the disagreement below), so
there is no clean branch to carve out. The blocking shape is factory-wide, not one branch among
siblings.

> ### ⚠ My code evidence does not reproduce the blocking column — please reconcile
>
> **The gate stands and I have not altered it.** `Is able to port?` is a derived cell that this audit
> reads rather than vets, and `Smuggled pointer` is not one of the columns the recipe asks me to
> cross-check. But the evidence is specific enough to be worth routing, because it bears directly on
> whether this factory is near-ready or genuinely broken.
>
> **What I found: on the descriptor path, no raw buffer address reaches any kernel.** Both descriptor
> builders assemble runtime args as a plain `std::vector<uint32_t>` containing *placeholder*
> `.address()` values, then convert to a
> `std::vector<std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>>` and **overwrite
> every address slot with the tensor reference** before calling `emplace_runtime_args`. All five
> address-carrying emplace sites go through that rebinding:
>
> | Emplace site | Rebinds |
> |---|---|
> | `:4106` (mcast_in0, in0 sender) | `in0_sender_variant[0] = in0_tensor` (`:4105`) |
> | `:4193` (mcast_in0, in1 sender/writer) | `[0] = in1_tensor`, `[7] = out_tensor`, `[18] = *bias_tensor` (`:4188-4191`) |
> | `:5062` (mcast_in1, in1 sender/writer) | `[0] = in1_tensor`, `[7] = out_tensor`, `[18] = *bias_tensor` (`:5057-5060`) |
> | `:5110` (mcast_in1, in1 receiver/writer) | `[2] = out_tensor` (`:5109`) |
> | `:5133` (mcast_in1, in0 sender) | `[0] = in0_tensor` (`:5132`) |
>
> The only three emplace sites that skip the variant form — `:4072`, `:4074`, `:4076` — push a vector
> (`:4057-4065`) that contains **no address at all**: a core index, mcast NOC start/end coordinates,
> and the mcast noc_x / noc_y lists. That is the in0-**sharded** branch, where in0 arrives through a
> globally-allocated CB rather than an address argument.
>
> So every `.address()` call in the descriptor paths (`:4083`, `:4122`, `:4135`, and the bias one at
> `:4170`) is a dead placeholder value, overwritten before dispatch.
>
> **Scan completeness.** The claim covers every argument channel, not just `runtime_args`: the
> descriptor paths declare **no** common runtime args, bake **no** address into a compile-time arg,
> and pass **no** semaphore address as an argument. The four `.address()` / `->address()` sites
> (`:4083`, `:4122`, `:4135`, `:4170`) are the complete set, and all four are rebound.
>
> **Where the file's genuine smuggled pointers are.** Classifying all 25 address expressions in the
> file by enclosing function and annotation status:
>
> | Location | Sites | Annotation | Rebound? | Reaches a kernel raw? |
> |---|---|---|---|---|
> | **Sibling factory** — `process_*` builders + `override_*` helpers (`:1075`-`:3115`) | **21** | 19 un-annotated, 2 annotated | **no** | **yes** |
> | **This factory** — `create_program_mcast_in0_descriptor` | 4 | 3 un-annotated, 1 annotated | **yes** (`:4105`, `:4188`, `:4189`, `:4191`) | no — dead placeholders |
> | **This factory** — `create_program_mcast_in1_descriptor` | 0 | — | — | no |
>
> The 21 genuine sites all belong to the **sibling** factory: raw addresses written into
> `std::vector<uint32_t>` args with no rebinding, re-patched only by the void
> `override_runtime_arguments` that factory does call. (`fused_op_signaler` is hardcoded
> `std::nullopt` at `:5609`, so the CCL signaler's `push_matmul_fused_op_rt_args` branches contribute
> no arguments on this factory's path.)
>
> **Explanations, ordered by what the evidence supports.** Git history rules out the most obvious
> one, so it is listed last rather than first:
>
> 0. **Most likely: a true positive on the pattern, a false positive on the hazard — in this
>    factory's own code.** `:4083`, `:4122` and `:4135` are **un-annotated** raw `->address()` values
>    inside runtime-arg construction, which is exactly what the column's definition ("an un-annotated
>    pointer argument") describes and what any sweep would flag. What makes them benign is the
>    variant rebinding 20-100 lines below. They are un-annotated precisely *because* they are
>    harmless — a `smuggled-rta-ok` marker would misdescribe them. Only the bias site (`:4170`)
>    carries the marker, plausibly because its rebinding is conditional on `bias_tensor.has_value()`
>    and so reads as more reachable. On this explanation the cell is **correct as defined** and the
>    definition simply does not distinguish a rebound placeholder from a live stale address.
> 1. **The column may additionally be attributed at file or device-op granularity.** The sibling factory's
>    legacy paths **in the same `.cpp`** do carry raw address RTAs with no variant rebinding —
>    `:1075`, `:1114`, `:1127` (mcast_in0 legacy), `:2003`, `:2016`, `:2058`, `:2096` (mcast_in1
>    legacy), `:2785` (gather_in0) — patched only through the void `override_runtime_arguments`.
>    **Both** matmul mcast-1d factory rows read `Smuggled pointer = yes`, which is what a human
>    classifying by reading this 5,885-line file would produce.
> 2. **The `smuggled-rta-ok` convention may *be* the classification.** If the column is driven by a
>    repo-wide sweep for `->address()` in runtime-arg construction, then `yes` correctly records that
>    *the pattern is present*, with the `-ok` suffix meaning reviewed-and-accepted rather than
>    absent. On that reading the cell is not wrong — it is answering a narrower question than "does a
>    stale address reach a kernel," and the two answers legitimately differ here.
> 3. **The classification predates a remediation — evidence argues against this.** It was the natural
>    first reading, but the dates do not support it. The variant rebinding landed **2026-05-29**
>    (`[MeshTensor Integration] Matmul`, #44220), and the descriptor migration before it landed
>    **2026-05-07** (#43578, titled *"Migrate matmul factories to create_descriptor() with cache-hit
>    buffer patching"*). The `// smuggled-rta-ok` marker on `:4170` was added **2026-07-08** (#47773)
>    — six weeks *after* the rebinding already existed. Someone marked that line as a smuggled RTA
>    with the fix in place, which is the opposite of a stale classification.
>
> **Routing:** a question to the **readiness-sheet owner** (and TTNN), not a defect allegation. The
> question to ask is narrow: *what does `Smuggled pointer` record — the presence of the textual
> pattern, or a stale address actually reaching a kernel — and is it scoped per factory row or per
> file?* The answer determines whether this factory is near-ready or genuinely needs a rewrite, which
> is why the port-work detail below was produced rather than deferred.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreReuseMcast1DProgramFactory`) reads **`no`**,
  with `Op Classification = Broken Op`, `Diego validation = no`, `Porting Target = (N/A)`. The
  blocking column is `Smuggled pointer = yes` → routed to **TTNN**, who fix the op first.

  **Lightweight cross-check — clean on every primary column.** No spreadsheet-broken finding.

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returning `ProgramDescriptor` (hpp:35) | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op — see below | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across the op's `.cpp` / `.hpp` | ✓ |
  | `Override runtime args method?` | `yes` | Present at `…mcast_1d_program_factory.cpp:5482` (hpp:28) | ✓ |
  | `Pybind descriptor` | `PR` | Present at `matmul_nanobind.cpp:1277-1290` | ✓ (in-flight PR; still on this checkout) |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1 — no phantom or missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` row, and
  `Op-owned tensors?` is empty (only ever `yes` on `WorkloadDescriptor`).

  **On the custom hash.** The device-op declares `compute_descriptor_program_hash`
  (`device/matmul_device_operation.hpp:50`) with a comment that it is *deliberately* not named
  `compute_program_hash`, so the framework does **not** detect a custom cache hash; it is reached
  only through a pybind alias. The framework uses the **default reflection hash**, which is what the
  sheet's `no` records. Nothing here for a port to touch.

- **Device 2.0 (every kernel used): GREEN.** All six kernels reachable from this factory are
  structurally Device 2.0 — `Noc` from `noc.h`, `DataflowBuffer` wrappers, `TensorAccessor`. Zero
  hits for broad Device-1.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`,
  raw `noc_async_read(` / `noc_async_write(`, raw `noc_semaphore_*`) and no non-sanctioned CB-index
  free-function holdovers. All six are matmul-owned, so there is no donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Path(s) | Device 2.0 |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | both | ✓ |
  | `dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | mcast_in0 (in0 sharded) | ✓ |
  | `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | mcast_in0 | ✓ |
  | `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | both | ✓ |
  | `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | mcast_in1 | ✓ |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | both | ✓ |

  Five of the six call `get_local_cb_interface(...)` and/or `get_tile_size(...)`. Both are on the
  recipe's **sanctioned** list, which "does not turn on what object is in scope," so they stay
  sanctioned even where a `DataflowBuffer` is in scope. Not Device 2.0 violations; they are Metal 2.0
  port-stage rewrites (onto the object, or kept in free-function form with the binding token where
  the value is `constexpr`).

- **Feature compatibility: GREEN.** Every Appendix A entry scanned against the factory's reachable
  code (3141-5150) and its six kernels. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | **Zero GCB signals in the descriptor paths** — no type reference, no 4-arg `experimental::CreateCircularBuffer`, no `remote_index`, no `remote_cb_config`, no `global_cb` parameter. This is the sharp difference from the sibling factory sharing this file, whose legacy paths do use one; factory selection at `matmul_device_operation.cpp:2204` routes any GCB-backed config *away* from this factory, with the in-code reason *"ProgramDescriptor cannot attach an experimental GlobalCircularBuffer."* |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress` in any form, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`. The factory declares two ordinary `SemaphoreDescriptor`s per path (`:4012-4015`, `:4958-4961`) |

- **Offset base pointers: GREEN.** No host-side fold anywhere. A scan for `address() +`, `addr + …`
  and `+ …offset…` across the file returns **zero** hits, and in the descriptor paths the four
  `.address()` sites are placeholder values overwritten by tensor references before dispatch (see
  the disagreement box). No Type 1, no Type 2, no Type 3, no Type 4. The checked-in offset triage
  lists no matmul row, and my own scan agrees.

- **TensorAccessor 3rd argument: N/A.** No accessor in any of the six kernels passes a third
  (page-size) argument — every construction is 2-arg. The subject never fires.

---

## Informational subjects — **run, not skipped**

The Red-outcome scoping rule asks which side the RED clears on, and this is a case where it plausibly
clears **with the op's code untouched**: the most likely resolution of the disagreement above is a
readiness-sheet reclassification, not a rewrite. On that side the recipe is explicit — re-audit would
read the *same* code, so today's detail survives intact and deferring costs a second full pass while
saving nothing.

I judged it that way deliberately. If TTNN instead rewrites the factory, some of the detail below
goes stale — that is the cost I accepted, and it is bounded. The upside is that if the gate lifts
without a code change, a port can start immediately from this report.

Note also the recipe's own steer: a concept-gate failure on a **`descriptor`** op is *not*
automatically a no-subset case and "usually means auditing it in full." That is what this is.

---

## Port-work summary  *(for when the gate lifts — no brief is issued on a RED)*

- **Target concept: `CustomProgramSpecFactoryConcept`** (`Override runtime args method? = yes`), and
  **the existing override has the wrong shape, not just the wrong body.** It is declared
  `static void override_runtime_arguments(Program&, const shared_variables_t&, …)` (hpp:28,
  cpp:5482). Two problems the port must solve together:
  1. The custom concept requires an override **returning `ProgramRunArgs`**; only the return type is
     concept-enforced, so a void one silently leaves the factory on the base concept.
  2. Its second parameter is `shared_variables_t` — kernel handles, CB handles and core lists — which
     `create_descriptor` **never produces**. On the descriptor path there is nothing to populate it
     from, so the method as written cannot be driven by the descriptor adapter at all. It exists for
     the sibling MeshWorkload factory and for external CCL fused-op callers
     (`all_gather_matmul_async`, `matmul_reduce_scatter_async`), which supply that state themselves.

  The port therefore *translates* the override into a `ProgramRunArgs`-returning form keyed by
  binding name rather than by handle — and must not simply delete it, which would drop the op to the
  base concept and silently discard every non-tensor refresh.

- **Tensor bindings — four, all Case 1.** `in0`, `in1`, `output`, and (conditionally) `bias`. Each is
  already delivered as a `MeshTensor` reference through the variant mechanism, and each is consumed
  kernel-side through a `TensorAccessor`. Straight translation to `TensorParameter` / `TensorBinding`;
  the address slots and their `TensorAccessorArgs` plumbing both disappear. No Case 2 site, so the
  `get_bank_base_address` bridge is not needed anywhere.

- **CB endpoints.** Two semaphores per path and a CB set that is mostly plain 1:1 (in0 reader
  produces `c_0`, compute consumes; in1 sender produces `c_1`, compute consumes; compute produces
  `c_4`, the writer consumes). Three dispositions need real attention:

  - **An aliased-DFB group of two *or three* members, conditional on config.** When output and
    intermediate share a buffer, one `CBDescriptor` carries `c_4` **and** `c_5` (`:3951-3975`,
    `:4903-4924`); when `bias_reload_alias` is additionally set it carries a **third** index,
    `cb_intermed0_alias = tt::CBIndex::c_7` (`:3799`, `:3942`, `:3967`, `:4892`, `:4917`) — described
    in-code as *"a second buffer index over the same SRAM"* marked `UnpackToDestFp32` for the bias
    reload. Port as two or three `DataflowBufferSpec`s whose `advanced_options.alias_with` forms a
    **strict clique** (every member naming every other), all the same total size and bound to the
    same kernels. In the non-shared branch (`:3916-3947`, `:4866-4900`) they are separate descriptors
    and no aliasing applies. **The group size is config-dependent — derive it per instantiation.**
  - **`c_6` is a 32-byte scratch CB** (`:3899-3907`, mcast_in0 only), commented *"Local L1 to store
    temp vars"*, with no tensor backing. Run the toucher census on it specifically: a single-toucher
    result is a **self-loop** (bind the one kernel PRODUCER + CONSUMER), a zero-toucher result is a
    **dead-CB drop**. It is the one CB here most likely to fall outside the plain 1:1 shape.
  - **Borrowed-memory CBs are used in both paths** — `c_1 ← in1_tensor` (`:3879`), `c_2 ← in0_tensor`
    (`:3895`, `:4843`), `c_0 ← in0_tensor` (`:4828`), `c_4 ← out_tensor` (`:3925`, `:3973`, `:4875`,
    `:4923`), each conditional on the corresponding operand being sharded. Each becomes
    `DataflowBufferSpec::borrowed_from` naming the matching `TensorParameter`.

- **A CB index travels through a preprocessor define.** `mm_kernel_defines["MM_PARTIALS_RELOAD_ALIAS_CB"]`
  is set to the numeric value of `cb_intermed0_alias` (`:3803`, `:4765`), conditional on
  `bias_reload_alias`. In Metal 2.0 a CB index becomes a `DFBBinding`, never a scalar — so this needs
  the **conditional-binding pattern**: bind the DFB conditionally, emit a matching flag via
  `KernelSpec::compiler_options.defines`, and `#ifdef`-gate the kernel-side alias and its uses.

- **`unpack_to_dest_mode` → `unpack_modes`, with all three hazards live.** The factory sets
  `.unpack_to_dest_mode` on its `ComputeConfigDescriptor` (`:3842`). The translation must
  **reindex** (legacy `vector<UnpackToDestMode>` by CB id → `Table<DFBSpecName, UnpackMode>` by
  name), **translate values without inverting them** (`UnpackToDestFp32` → `UnpackMode::UnpackToDest`;
  `Default` → `UnpackToSrc`, normally expressed by omission), and satisfy the **newly-required
  explicit entry** for any Float32 DFB a compute kernel consumes with `enable_32_bit_dest` on —
  which is reachable here, since `interm0_data_format` becomes `Float32` when `fp32_dest_acc_en` is
  set (`:3200-3202`). The `c_7` alias exists precisely to carry a different unpack mode over the same
  SRAM, so its entry is load-bearing. A conditionally-bound DFB's entry must be gated on the same
  condition as its binding.

- **⚠ A resolved compute-config field is dropped, and the port must preserve that.**
  `create_descriptor` resolves the full five-tuple at `:5558-5559` — including `dst_full_sync_en` —
  but **neither** descriptor builder takes it as a parameter (`:3141-3147`, `:4217-4223`) and neither
  sets it on the `ComputeConfigDescriptor` (`:3839-3843`). The resolved value is silently discarded
  and the descriptor default applies, so this op ignores the knob whatever the caller passes.

  This is the dropped-field case the recipe says to check every time, and it fires here. The TTNN
  helper `to_compute_hardware_config` reads the *resolved* config and would hand the caller's value
  back — changing behavior. The port must explicitly pin `double_buffer_dest` to the legacy-default
  result (`dst_full_sync_en = false` → **`double_buffer_dest = true`**) on the returned config. This
  is **preserved behavior, reported not repaired** — do not "fix" the ignored parameter.

- **`opt_level` — absent.** `grep -n opt_level` over the descriptor paths returns nothing. An unset
  `KernelDescriptor::opt_level` still resolves to the legacy per-kernel-type default — **`O3` for a
  `ComputeConfigDescriptor`**, `O2` for DM — while Metal 2.0's `CompilerOptions` defaults to `O2` for
  both. Set `O3` explicitly on every compute `KernelSpec` the port builds.

- **DM configs are custom — replicate, do not use the helpers.** Both paths build explicit
  `DataMovementConfigDescriptor`s with `RISCV_1 / in0_noc` and `RISCV_0 / in1_noc` (`:3675`, `:3696`,
  `:3716`, `:3731`, `:3748`, `:4676`, `:4692`, `:4709`), where the NOCs come from the factory's own
  `get_preferred_noc` helper rather than the reader/writer defaults. Copy each field verbatim into a
  `DataMovementGen1Config`; reaching for `create_reader_datamovement_config` /
  `create_writer_datamovement_config` would substitute the default triple and regress silently.

- **RTA varargs: none — and there is a trap here.** `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`
  reads roughly a dozen arguments through a running `rt_args_idx++` counter (lines 18-35). **These are
  all named args, not varargs.** Each is a distinct field read exactly once — `in1_mcast_sender_noc_x`,
  `out_tensor_addr`, `out_num_nonzero_subblocks_h`, and so on — in a block at the top of the kernel.
  A running counter is not a vararg signal; it appears in both fixed reads and genuine loops. No
  kernel here reads arguments in a loop or at a data-computed index, so nothing justifies
  `get_vararg`. This is the silent error the recipe flags as trap (1).

- **Device-operation-class edits the port forces** — two sanctioned exceptions:
  1. **Remove the pybound factory entry point.** `matmul_nanobind.cpp:1277-1290` is an
     `nb::class_<ttnn::prim::MatmulMultiCoreReuseMcast1DProgramFactory>` block whose only member is
     `create_descriptor`. Deletion is mandatory once that method goes; user-visible API change.
  2. **Drop the pybind-hook-only parameter.** `create_descriptor`'s fourth argument,
     `const std::optional<CoreRangeSet>& core_range_set`, is **ignored by the factory body** —
     spelled `/*core_range_set*/` at `:5496`. Drop it; nothing reads it.

  Exception 3 does not apply — the op has a proper `program_factory_t` variant.

---

## Heads-ups

- **Every one of the six kernels is shared, and none has a fork yet.** This is the largest structural
  constraint on the eventual port. All six live in matmul's own directory, so this is the *intra-op*
  shared-kernel case — including, critically, sharing with the **sibling factory in the very same
  `.cpp`**, which binds the same sources through its legacy build path and is itself blocked
  (GlobalCircularBuffer), so it cannot co-port.

  | Kernel | Other binding factories | `_metal2` fork? | Rung |
  |---|---|---|---|
  | `in0_sender_padding` | Mcast2D, Sparse, **MeshWorkload sibling** | no | **2 — create** |
  | `in0_sender_receiver_padding_block_sharded` | Mcast2D, **MeshWorkload sibling** | no | **2 — create** |
  | `in0_receiver` | Mcast2D, Sparse, **MeshWorkload sibling** | no | **2 — create** |
  | `in1_sender_writer_padding` | Mcast2D, Sparse, **MeshWorkload sibling** | no | **2 — create** |
  | `in1_receiver_writer_padding` | Mcast2D, **MeshWorkload sibling** | no | **2 — create** |
  | `bmm_large_block_zm_fused_bias_activation` | 5 others (Optimized, McastDRAMSharded, Mcast2D, BatchedHS, Sparse) | no | **2 — create** |

  The rung-1 check was run **locationally** on `device/kernels/dataflow/` and
  `device/kernels/compute/`: no `_metal2` sibling exists for any of them. So a port of this factory
  creates **six forks**, and whatever binding vocabulary the first one uses becomes the interface
  every later consumer inherits. That is worth agreeing across the matmul factories before any of
  them ports, rather than settling it inside whichever port happens to land first.

- **The two descriptor paths bind overlapping but different kernel sets.** `mcast_in0` uses the in0
  sender / block-sharded sender-receiver / in0 receiver / in1 sender-writer; `mcast_in1` uses the in0
  sender / in1 sender-writer / in1 receiver-writer. Both use the shared compute kernel. Map each
  DFB's producer and consumer **per path** — a role that holds on one path need not hold on the
  other.

- **`bias_reload_alias` gates real structure, not just a value.** It controls whether the alias group
  has two members or three, whether the `MM_PARTIALS_RELOAD_ALIAS_CB` define is emitted, and which
  CB the bias reload copies through (`:3836`, `:4798`). Treat it as a configuration axis when
  classifying CBs, not as a detail.

---

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. No function-call escapes to
  another op's helpers and no file-path escapes — all six kernels are owned and instantiated by
  matmul. Includes are `api/*` (LLK / HAL, donor class 1) plus in-op siblings. **Borrowed kernel
  files: none.**

- **Relaxation candidates:** none. No custom hash to mine; the sheet's `TensorParameter relaxation`
  is `none`.

- **TTNN factory analysis:** op-owned tensors — none. MeshWorkload need — none (`Execution Model =
  SPMD`, `Concept = descriptor`). Pybind `create_descriptor` — `matmul_nanobind.cpp:1277-1290`.
  Other risky pybind — the device-op class binding at `matmul_nanobind.cpp:1222-1237`, which survives
  the port untouched. Custom hash — none framework-visible. `get_dynamic_runtime_args` — absent.
  `override_runtime_arguments` — present at `:5482`, returning void, with the shape problem described
  in Port-work. Target concept — `CustomProgramSpecFactoryConcept`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead placeholder `.address()` calls.** `:4083`, `:4122`, `:4135` and `:4170` compute a buffer
  address that is immediately overwritten by the variant rebinding a few dozen lines later. Harmless,
  but they are what a text search for smuggled pointers finds first, and they cost a reader real time
  before the rebinding turns up. A comment at the construction site (the bias one already has one)
  or dropping the placeholder to `0u` would remove a standing trip hazard.

- **A resolved-then-discarded compute-config field.** `dst_full_sync_en` is resolved at `:5558` and
  never reaches either descriptor builder, so the op ignores it whatever a caller passes. Not a port
  concern beyond preserving it (see Port-work), but the ops team may want to know the knob is inert
  on this factory.

- **`create_descriptor`'s `core_range_set` parameter is accepted and ignored** (`/*core_range_set*/`
  at `:5496`), while the sibling `MatmulMultiCoreReuseOptimizedProgramFactory` genuinely uses its
  equivalent. A Python caller passing one here has it silently discarded. The port removes the
  parameter, so this resolves itself.

---

## Questions for the user

1. **Please reconcile the `Smuggled pointer = yes` classification** against the evidence in the
   disagreement box above. The gate holds either way, but the answer determines whether this factory
   is one sheet edit away from portable or genuinely needs a TTNN rewrite — and therefore whether the
   port-work detail in this report is usable as-is.

2. **Fork vocabulary should be agreed before any matmul factory ports.** Six shared kernels have no
   `_metal2` fork yet, and five of the six are bound by the Mcast2D and sparse factories as well.
   Whichever factory ports first sets names the others cannot change. Worth one decision now.

3. **Scope.** Six factories in this directory remain unaudited, plus the sparse device-op's factory.
   Should those follow, and in what order?

## Recipe notes

- **The "read the derived cell, don't vet it" rule was load-bearing and I think correctly drawn.**
  This audit produced concrete evidence against a blocking column, and the rule kept that out of the
  verdict while the report structure still carried it to the right owner. Without the rule I would
  have been tempted to GREEN the factory on my own reading, which would have been wrong — I cannot
  see the derivation, and explanation (2) in the box is entirely plausible.

- **Suggestion: the cross-check list could say what to do about a *non*-cross-checked column that
  your evidence contradicts.** `Smuggled pointer` is a blocking column but is not on the list of
  columns the auditor verifies, so when code evidence contradicts it there is no prescribed route.
  I treated it as a question to the sheet owner by analogy with the unattributed-`no` case. A
  sentence confirming that (or directing otherwise) would remove the judgement call.

- **Friction — the shared-kernel census is expensive when two factories live in one file.** The raw
  binder count for five of the six kernels includes
  `matmul_multicore_reuse_mcast_1d_program_factory.cpp` itself, because the sibling factory shares the
  file. Distinguishing "another factory in the same file binds this" from "my own factory binds this"
  needed per-function line-range analysis rather than a file-level grep. The recipe's intra-op case
  covers the situation, but its worked framing assumes one factory per file.
