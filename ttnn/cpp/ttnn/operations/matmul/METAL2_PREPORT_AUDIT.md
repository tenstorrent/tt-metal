# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMultiCoreReuseOptimizedProgramFactory`.**
The op directory holds two DeviceOperations and eight ProgramFactories; this audit covers a single
factory by request. The other seven are named below for disambiguation only and were **not**
audited — no statement in this report is a verdict on any of them.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMultiCoreReuseOptimizedProgramFactory`** ← **audited**
    (`device/factory/matmul_multicore_reuse_optimized_program_factory.{hpp,cpp}`)
  - `MatmulMultiCoreProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast2DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` — not audited
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

> **Structure.** The `.cpp` (637 lines) holds exactly two function bodies:
> `default_core_range` (31-34) and `create_descriptor` (36-635). No legacy sibling builder, no
> `override_runtime_arguments`, no helper exported for CCL fused ops. Every finding below is
> unambiguously in scope.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns);
exactly one row matches this factory.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **GREEN** (for `MatmulMultiCoreReuseOptimizedProgramFactory` only) |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMultiCoreReuseOptimizedProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 3 bound kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | **Ok** — all three kernels are matmul-owned; no donor function-call escapes |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A · N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** (framework-visible) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — absent from the factory entirely |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — `matmul_nanobind.cpp:1240-1258`, **plus a second bound member** (`default_core_range`) |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (base) |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere in the factory |
| *Port work* — Tensor bindings (per binding) | 4 bindings, all **Case 1** |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — every construction is 2-arg |
| *Port work* — CB endpoints | 2 compute self-loops, an aliased pair, rest plain 1:1; **zero semaphores** |

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, alongside this file).

Every gate cleared. The factory itself is among the simplest in the op — no multicast, **zero
semaphores**, two data-movement kernels rather than three, and one program builder — but it carries
**one finding with a blast radius beyond the op**, which the porter must not discover mid-conversion:

> **This factory's pybound `create_descriptor` has a live, in-tree Python consumer.**
> `models/experimental/ops/descriptors/matmul.py:120` calls
> `factory.create_descriptor(operation_params, tensor_args, [out], core_range_set)` on whichever
> factory `ttnn.matmul_select_program_factory(...)` returns — and this factory is selectable.
> The class is additionally **exported into the public `ttnn` namespace** (`ttnn/__init__.py:541`,
> aliased in `ttnn/ttnn/operations/matmul.py:25`), which no other matmul *factory* is.
>
> Unlike the recipe's canonical case, the `core_range_set` parameter is therefore **not** a
> vestigial hook artifact: production C++ never sets it, but a checked-in Python descriptor framework
> does, and that framework drives further machinery (`op_descriptor.py`, `fusion/fusion.py`). See
> [Heads-ups](#heads-ups) — this needs a decision, not a porter judgement call.

There is no blocked code path within the factory and no subset to carve.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreReuseOptimizedProgramFactory`) reads **`yes`**,
  with `Diego validation = yes`, `Op Classification = PD Op (pointer-patching)`,
  `Execution Model = SPMD`, `Porting Target = ProgramSpecFactoryConcept`. Two non-gating cells to
  carry forward: `Pointer patching perf issue? = "suspect perf regression (+ fixed latent bug)"` and
  `Formerly custom hashed? = "yes"` — both refer to the earlier ProgramDescriptor migration, not to a
  Metal 2.0 port. Recorded so a post-port performance comparison against an older baseline is not
  misread.

  Lightweight cross-check clean on every primary column:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returning `ProgramDescriptor` (hpp:13-17) | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op — see below | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across the op's `.cpp` / `.hpp` | ✓ |
  | `Override runtime args method?` | `no` | Zero hits for `override_runtime_arguments` in the factory | ✓ |
  | `Pybind descriptor` | `PR` | Present at `matmul_nanobind.cpp:1240-1258` | ✓ (in-flight PR; still on this checkout) |
  | `Smuggled pointer` | `no` | **Independently verified** — the factory contains **no `.address()` / `->address()` expression at all**; tensors are pushed into `emplace_runtime_args` as objects (`:459`, `:464-475`) | ✓ |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1 — no phantom or missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` row, and
  `Op-owned tensors?` is empty (only ever `yes` on `WorkloadDescriptor`).

  **On the custom hash.** The device-op declares `compute_descriptor_program_hash`
  (`device/matmul_device_operation.hpp:50`) with a comment that it is *deliberately* not named
  `compute_program_hash`, so the framework does not detect a custom cache hash; it is reached only
  through a pybind alias — which `models/experimental/ops/descriptors/matmul.py:105` in fact calls as
  `ttnn.MatmulDeviceOperation.compute_program_hash`. The framework itself uses the **default
  reflection hash**. Nothing for the port to touch.

- **Device 2.0 (every kernel used): GREEN.** All three bound kernels are structurally Device 2.0 —
  `Noc` from `noc.h`, `DataflowBuffer` wrappers, `TensorAccessor`. Zero hits for broad Device-1.0
  idioms (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, raw `noc_async_read(` /
  `noc_async_write(`, raw `noc_semaphore_*`) and no non-sanctioned CB-index free-function holdovers.
  All three are matmul-owned, so there is no donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Role | Device 2.0 evidence |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0.cpp` | in0 reader (reader processor) | `Noc` ×5, `DataflowBuffer` ×1, `TensorAccessor` ×2 |
  | `dataflow/reader_writer_bmm_tile_layout_in1.cpp` | **reads in1 *and* writes the output** (writer processor) | `Noc` ×7, `DataflowBuffer` ×3, `TensorAccessor` ×6 |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | compute | `DataflowBuffer` ×11, LLK compute APIs |

  Note the two-DM-kernel shape: there is no separate writer kernel. `reader_writer_bmm_tile_layout_in1`
  does both the in1 read and the output write from the writer processor, leaving the reader processor
  to do nothing but in0.

- **Feature compatibility: GREEN.** Every Appendix A entry scanned against the factory and its three
  kernels. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_index` / `remote_cb_config`, no `global_cb` parameter |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress` in any form, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | The factory declares **zero semaphores of any kind** — there is no cross-core synchronisation to express |

- **Offset base pointers: GREEN.** The factory contains **no `.address()` / `->address()` expression
  at all**, so there is no host-side arithmetic that could fold an offset into a base. Addresses
  reach the kernels as tensor objects pushed into the runtime-arg lists (`:459` for in0; `:464`,
  `:467`, `:470` for in1 / output / bias), which the framework resolves. No Type 1, no Type 2, no
  Type 3, no Type 4. The checked-in offset triage lists no matmul row, and my own scan agrees.

- **TensorAccessor 3rd argument: N/A.** No accessor in any of the three kernels passes a third
  (page-size) argument — every construction is 2-arg. The subject never fires.

---

## Port-work summary  *(mirrors the brief)*

- **Target concept: `ProgramSpecFactoryConcept`** (base). The factory has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and the port
  writes one method, `create_program_artifacts`. **Do not add an override.**

- **Tensor bindings — four, all Case 1.** `in0`, `in1`, `output`, and (conditionally) `bias`. Each
  is delivered as a tensor object through `emplace_runtime_args` and consumed kernel-side through a
  `TensorAccessor`. Straight translation to `TensorParameter` / `TensorBinding`; the address slots and
  their `TensorAccessorArgs` plumbing both disappear, so re-index what remains. No Case 2 site — the
  `get_bank_base_address` bridge is not needed anywhere.

- **CB endpoints — six CBs, three dispositions, and no semaphores.** Census run per node across all
  three kernels.

  | CB | Backing | Touchers | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` in0 | regular, or **borrowed** ← in0 when in0 is sharded | in0 reader (locked producer, `:48`/`:49`, `:65`/`:106`), compute (locked consumer) | plain 1:1 | `borrowed_from` when sharded |
  | `c_1` in1 | regular, or **borrowed** ← in1 when in1 is sharded | in1 reader/writer (locked producer, `:105`/`:106`, `:128`/`:151`), compute (locked consumer) | plain 1:1 | `borrowed_from` when sharded |
  | `c_3` bias | regular | in1 kernel (locked producer, `:86`/`:100`), compute (locked consumer) | plain 1:1 | **conditional binding** (bias present) |
  | `c_4` out | regular, or **borrowed** ← output when output is sharded | compute (locked producer, packs), in1 kernel (locked consumer, `:166`/`:187`/`:198`) | plain 1:1 | `borrowed_from` when sharded |
  | `c_5` interm0 | regular | compute only | 1 toucher | **compute self-loop** |
  | `c_10` in0 transposed | regular, **conditional** on `in0_transpose_tile` | compute only — the transpose target it then reads back | 1 toucher | **compute self-loop**, conditionally bound |

  There are **no semaphores** to translate — this factory does no multicast, so each core reads its
  own operand blocks independently.

- **Aliased DFBs — a two-member group, config-dependent.** When
  `interm0_data_format == output_data_format` and not (`untilize_out` with more than one W subblock),
  a **single `CBDescriptor` carries both `c_4` and `c_5`** (`:614-624`) — two distinct buffers
  sharing one L1 region, with `output_cb_desc.tensor = output_is_sharded ? &output : nullptr`. Port
  as **two `DataflowBufferSpec`s with mutual `advanced_options.alias_with`** (same total size, same
  bound kernels, strict clique). In the other branch (`:598-607`) they are separate descriptors and
  no aliasing applies. **Derive per instantiation.** The group is exactly two members — there is no
  third alias index.

- **⚠ `packer_l1_acc_en` uses a threshold this factory does not share with its siblings.** Line 112
  reads `bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);` — **`> 2`**, where the other
  matmul factories use `> 1`. It feeds `interm0_data_format` (`:114`), which in turn decides whether
  the aliased-DFB branch is taken at all. **Carry the `> 2` across verbatim.** "Correcting" it to
  `> 1` would change both the intermediate format and the CB topology, and is exactly the kind of
  silent behaviour change the porting invariant forbids.

- **Preserved multiplicity — two compute KernelSpecs, not one.** The factory calls
  `split_work_to_cores` and emits two `ComputeConfigDescriptor`s of the same source, for
  `core_group_1` and `core_group_2` (configs at `:500-504` and `:543-547`), the second conditional on
  the group being non-empty. Port as **two KernelSpecs of the same source in two WorkUnitSpecs over
  disjoint node sets**, each binding the same DFBs with the same roles. Each node sees exactly one
  instance, so these are ordinary single-role bindings — **not** `allow_instance_multi_binding`.
  Demoting the per-group count to an RTA to collapse them is the documented anti-pattern.

- **`opt_level` — absent.** `grep -n opt_level` returns nothing. An unset
  `KernelDescriptor::opt_level` still resolves to the legacy per-kernel-type default — **`O3` for a
  `ComputeConfigDescriptor`**, `O2` for DM — while Metal 2.0's `CompilerOptions` defaults to `O2` for
  both. Set `O3` explicitly on **both** compute `KernelSpec`s. The two DM kernels need nothing.

- **Hardware config — Style A, and nothing dropped.** The factory resolves a TTNN
  `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:78-79`), and both compute descriptors
  set all four helper-covered knobs — `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`,
  `math_approx_mode` (`:500-504`, `:543-547`). So `to_compute_hardware_config(device->arch(), config)`
  translates faithfully with nothing to reapply by hand. `packer_l1_acc` is genuinely *consumed*
  (`:112`, see above) but has no Metal 2.0 counterpart; no action.

  **No `unpack_to_dest_mode` is set**, so there is no legacy table to reindex. The only reason an
  `unpack_modes` entry would be required is the Float32-DFB-with-`enable_32_bit_dest` rule — check
  the resolved config, not the tensor dtypes.

  Both DM kernels use plain `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`, so the
  arch-agnostic `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`
  helpers apply. Match on the **resolved triple**, not the role name — note that
  `reader_writer_bmm_tile_layout_in1` is configured as a *writer* even though it also reads.

- **RTA varargs: none — and there is a trap.** Both DM kernels read their arguments through a running
  `rt_args_idx++` counter (in0 reader `:17-20`, in1 reader/writer `:17-18`, and onward). **These are
  all named args.** Each is a distinct field read exactly once in a block at the top; a running
  counter is not a vararg signal, and no kernel here reads arguments in a loop or at a data-computed
  index. This is the silent error the recipe flags as trap (1).

- **Device-operation-class edits the port forces.** Two sanctioned exceptions apply, but **both are
  complicated here by the live Python consumer** — see Heads-ups before acting:
  1. **Remove the pybound factory entry point.** `matmul_nanobind.cpp:1242-1254` is the
     `create_descriptor` `def_static` inside an `nb::class_<…ReuseOptimizedProgramFactory>` block.
     That method vanishes at port time, so the binding must go.
  2. **Drop the pybind-hook-only parameter.** `create_descriptor`'s fourth argument,
     `const std::optional<CoreRangeSet>& core_range_set`. **Unlike the sibling factories, this one
     genuinely uses it** — `:219` `else if (core_range_set.has_value())` feeding `split_work_to_cores`
     at `:227`. Production C++ always passes `std::nullopt`, so the branch falls through to the
     `program_config.allowed_worker_cores` path; dropping the parameter therefore also means deleting
     that live branch, which is a larger edit than the equivalent in sibling factories.

  Exception 3 does not apply — the op has a proper `program_factory_t` variant.

- **The op's cache key is not an edit.** No framework-visible custom hash exists.

---

## Heads-ups

- **⚠ The pybind removal has a named, in-tree Python consumer — this is the finding to resolve first.**
  `models/experimental/ops/descriptors/matmul.py` is a live descriptor framework that:
  - selects a factory with `ttnn.matmul_select_program_factory(operation_params, tensor_args)` (`:98`),
    explicitly excluding only `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (`:28`, `:39`) —
    so **this factory is selectable**;
  - calls `factory.create_descriptor(operation_params, tensor_args, [out], core_range_set)` (`:120`);
  - calls `ttnn.MatmulDeviceOperation.compute_program_hash` (`:105`) and `compute_output_specs`
    (`:113`);
  - documents `core_range_set` as *"the single source of truth for core placement"*.

  It is reached from `models/experimental/ops/descriptors/__init__.py` and used by
  `op_descriptor.py:138` and `fusion/fusion.py:871,879`.

  Additionally, **this factory is exported into the public `ttnn` namespace**
  (`ttnn/__init__.py:541`; aliased at `ttnn/ttnn/operations/matmul.py:25`) — the only matmul
  *factory* so exported. The sheet's `Pybind descriptor = PR` indicates removal is tracked in an
  in-flight PR, so this may already be in hand; confirm before porting rather than assuming.

  **The recipe's exception-2 framing does not cleanly fit.** It describes a parameter that "production
  code never sets — it exists only so a pybind test/introspection hook can drive the factory."
  Production *C++* never sets it, but a checked-in Python framework does, and that framework is more
  than an introspection hook. This is a coordination question, not a porter judgement call.

- **A second bound member complicates the pybind deletion.** The `nb::class_` block binds **two**
  statics: `create_descriptor` (`:1242-1254`) and `default_core_range` (`:1255-1258`). Only the first
  vanishes with the port, so the sanctioned exception covers only that `def_static` — leaving the
  class block alive with `default_core_range` alone.

  But `default_core_range` has **no production C++ caller at all**: its only references are its own
  declaration (`hpp:19`), its definition (`cpp:31`), and the pybind. It exists to let Python compute a
  core range and hand it to `create_descriptor` — precisely the workflow the port removes. (Contrast
  `LayerNormMultiCoreProgramFactory::default_core_range`, which *is* used in production C++ at
  `layernorm_op_multi_core.cpp:281` and called from Python at
  `models/experimental/ops/descriptors/normalization/_utils.py:44`.)

  Leaving it orphaned is defensible; removing it exceeds the sanctioned exception. **Flag, do not
  decide** — it belongs with the question above.

- **Only one kernel is shared, and it needs the first fork.**

  | Kernel | Copies | Binders | `_metal2` fork? | Rung |
  |---|---|---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0.cpp` | 1 | **1** — this factory | no | **convert in place** |
  | `dataflow/reader_writer_bmm_tile_layout_in1.cpp` | 1 | **1** — this factory | no | **convert in place** |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | 1 | **6** | **no** | **rung 2 — create the fork** |

  The compute kernel's other five binders are `matmul_multicore_reuse_batched_hs_dram_sharded`,
  `matmul_multicore_reuse_mcast_dram_sharded`, `matmul_multicore_reuse_mcast_2d`,
  `matmul_multicore_reuse_mcast_1d` (a file hosting **two** factories), and the sparse device-op's
  factory. The rung-1 check was run **locationally**: `find` over `matmul/device/kernels/` returns
  **zero** `*_metal2*` files. **Name the fork's bindings for the kernel, not for this factory.**

- **`transpose_a` / `transpose_b` are a live configuration axis.** They flip the in0/in1 stride
  computations (`:267-272`) and gate the `c_10` transposed-in0 CB and its `in0_transpose_tile` CTA.
  Map the DFB set and roles per transpose setting; a topology derived with transpose off will miss
  `c_10` entirely.

- **The bias path is a full `[M, N]` block, not a row.** The factory sets `BIAS_FULL_BLOCK` and
  guards it with a `TT_FATAL` requiring `N == per_core_N` and `M == per_core_M_per_batch`. That
  guard lives inside the factory body, so it is in the port's writeable surface — **carry it across
  verbatim**; the `TT_FATAL` census will flag it if it goes missing.

---

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** on the kernel side. No
  function-call escapes to another op's helpers and no file-path escapes — all three kernels are
  owned and instantiated by matmul. **Borrowed kernel files: none.** No C++ code outside the op calls
  into this factory (no override, no exported build helper).

  The coupling that *does* exist is on the **Python** side and runs inward: see Heads-ups.

- **Relaxation candidates:** none. No custom hash to mine; the sheet's `TensorParameter relaxation`
  is `none`.

- **TTNN factory analysis:** op-owned tensors — none. MeshWorkload need — none. Pybind
  `create_descriptor` — `matmul_nanobind.cpp:1242-1254`, plus a second bound member
  `default_core_range` at `:1255-1258`. Other risky pybind — the device-op class binding at
  `matmul_nanobind.cpp:1222-1237` (which exposes the renamed hash and *is* called from Python), and
  `ttnn.matmul_select_program_factory`, on which the Python descriptor framework's dispatch depends.
  Custom hash — none framework-visible. `get_dynamic_runtime_args` — absent.
  `override_runtime_arguments` — absent. Target concept — `ProgramSpecFactoryConcept`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **`default_core_range` is dead C++.** Declared and defined, never called from C++, exposed only
  through pybind. Worth an owner deciding its fate alongside the pybind question — it is not
  obviously wrong, just unreferenced from the language it lives in.

- **The `packer_l1_acc` threshold diverges from every sibling.** `num_blocks > 2` here (`:112`)
  versus `> 1` elsewhere in the op. It may well be deliberate — this factory has no multicast and a
  different reuse profile — but the divergence is undocumented at the line, and the value changes
  `interm0_data_format` and hence the CB topology. Worth a comment either way.

- **A DM kernel configured as a writer also reads.** `reader_writer_bmm_tile_layout_in1.cpp` reads
  in1 *and* writes the output, under a `WriterConfigDescriptor`. Correct and deliberate, but the
  name/role mismatch is a trip hazard for anyone matching hardware configs by role rather than by
  resolved triple.

---

## Questions for the user

1. **How should the Python descriptor consumer be handled?** `models/experimental/ops/descriptors/matmul.py`
   calls this factory's `create_descriptor` with a `core_range_set`, and the class is exported in the
   public `ttnn` namespace. Removing the pybind breaks that path for this factory. Options, none of
   which the porter should choose alone: land the removal with the in-flight `Pybind descriptor` PR
   that the sheet references; extend the framework's existing `_UNSUPPORTED_FACTORY` mechanism to
   exclude ported factories; or defer this factory until the descriptor framework is retired or
   migrated. Related: what becomes of `default_core_range`, which exists only to serve that workflow?

2. **Fork vocabulary for `bmm_large_block_zm_fused_bias_activation.cpp`.** This is the fourth
   consecutive audit whose port would create the first `_metal2` fork of a kernel bound by six
   factories. Whichever ports first fixes the binding names for the other five. Worth settling
   centrally rather than by whoever happens to go first.

3. **Scope.** Six factories in this directory remain unaudited, plus the sparse device-op's factory.
   Should those follow, and in what order?

## Recipe notes

- **Exception 2's framing assumes a narrower world than this op inhabits.** The recipe describes the
  pybind-hook-only parameter as one "production code never sets," with layernorm's `core_range_set`
  as the example. Here the same parameter is set by a checked-in Python framework that dispatches
  through `select_program_factory` and drives fusion machinery — so "production" depends on whether
  `models/experimental/` counts. The exception still points the porter the right way (drop the
  parameter, the port removes the workflow), but the *blast radius* it implies is too small, and the
  recipe gives no guidance on finding Python consumers beyond "record it in the report." A line
  suggesting a `--include=*.py` grep for the factory's exported name would have surfaced this in
  seconds; I found it only because I widened the search on my own initiative.

- **The dead-C++-but-pybound method has no home in the recipe.** `default_core_range` vanishes in
  usefulness with the port but not in compilability, so neither exception 1 (delete what references a
  vanished symbol) nor the scope discipline (leave everything else) resolves it cleanly. A sentence
  covering "a helper that exists only to feed the removed entry point" would close the gap.

- **Confirmation that the `> 2` / `> 1` style of divergence is worth an explicit check.** The
  compute-config section tells the porter to diff resolved values against the legacy config, which
  catches the `hw_config` fields — but `packer_l1_acc_en` is a *derived local*, not a config field,
  and it silently steers the CB topology. Nothing in the recipe would have flagged it; I caught it
  only by reading the derivation. A note that derived locals feeding CB shape deserve the same
  before/after scrutiny as `hw_config` would be worth adding.
