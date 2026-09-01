# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMultiCoreProgramFactory`.** The op directory holds two
DeviceOperations and eight ProgramFactories; this audit covers a single factory by request. The
other seven are named below for context only and were **not** audited.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMultiCoreProgramFactory`** (`device/factory/matmul_multicore_program_factory.cpp`) ← **audited**
  - `MatmulMultiCoreReuseOptimizedProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast2DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` — not audited
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **GREEN** (for `MatmulMultiCoreProgramFactory` only) |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMultiCoreProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 3 bound kernels are structurally Device 2.0 |
| *Prereqs* — Cross-op escapes | **Ok** — one class-4 donor (`kernel_helper_functions/pad_tile.hpp`), raw-`uint32_t` shape, crosses cleanly |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A · N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** (framework-visible). Renamed hook at `device/matmul_device_operation.hpp:50` — see Gate detail |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** (absent from the whole op) |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (this factory has none) |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — `matmul_nanobind.cpp:1260-1274` (not a gate; port deletes it) |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (base) |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere in the factory |
| *Port work* — Tensor bindings (per binding) | 3 bindings, all **Case 1** |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in this factory passes a 3rd argument |
| *Port work* — CB endpoints | **all legal** — 3 CBs, each a plain 1 producer + 1 consumer FIFO |

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, alongside this file).

Every gate cleared for `MatmulMultiCoreProgramFactory`. This is the op's simplest factory — no
reuse, no multicast, no semaphores, no sharding, no bias, three CBs and three kernels, all of them
matmul-private. There is no blocked code path within this factory and therefore no subset to carve.

**This verdict does not extend to the op's other seven factories.** The readiness sheet reports
`Is able to port? = no` for two of them (`MatmulMultiCoreReuseMcast1DProgramFactory`,
`MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — both classified `Broken Op`, both with a
smuggled pointer, the latter also carrying a `GlobalCircularBuffer`). Each remaining factory needs
its own audit before its own port.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreProgramFactory`) reads `yes`, with
  `Diego validation = yes`. Lightweight cross-check against the code came back clean on every
  primary column:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returning `ProgramDescriptor`, `matmul_multicore_program_factory.hpp:14` | ✓ |
  | `Custom hash` | `no` | no `compute_program_hash` on the device-op — see below | ✓ |
  | `Backdoor custom hash` | `no` | no `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | zero hits across `ttnn/cpp/ttnn/operations/matmul/` | ✓ |
  | `Override runtime args method?` | `no` | absent from the factory header and .cpp | ✓ |
  | `Pybind descriptor` | `PR` | present at `matmul_nanobind.cpp:1260-1274` | ✓ (see note) |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1, no phantom or missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` row, and
  `Op-owned tensors?` is empty (it is only ever `yes` on `WorkloadDescriptor`).

  **On the custom hash — the sheet is right and the code looks misleading.** The device-op declares
  `compute_descriptor_program_hash` (`device/matmul_device_operation.hpp:50`) with a comment stating
  it is *"intentionally NOT named compute_program_hash so the device-operation framework does NOT
  detect a custom program-cache hash"*, and that it is reached only through the pybind name
  `compute_program_hash` (`matmul_nanobind.cpp:1233-1237`). This is exactly the pybound-op rename
  the audit recipe anticipates: the framework uses the **default reflection hash**, so the sheet's
  `no` is correct, and the sheet's `Formerly custom hashed? = yes` corroborates it. A grep for
  `compute_program_hash` finds only the comment and the pybind name — not an override. **No
  discrepancy; nothing for the port to touch.**

  **`Pybind descriptor = PR`** means the removal is handled in an in-flight PR. On this checkout the
  binding is still present. Either way it is not a gate — the port deletes it (see Port-work).

- **Device 2.0 (every kernel used):** **GREEN.** All three bound kernels are structurally Device 2.0
  — `DataflowBuffer` wrappers, `Noc` from `noc.h`, `TensorAccessor`. No `InterleavedAddrGen` /
  `ShardedAddrGen` / raw `noc_async_*` / raw semaphore addresses anywhere.

  | Kernel | Role | Device 2.0 |
  |---|---|---|
  | `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp` | reader | ✓ `Noc::async_read`, `DataflowBuffer`, `TensorAccessor` |
  | `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | writer | ✓ `Noc::async_write`, `DataflowBuffer`, `TensorAccessor` |
  | `device/kernels/compute/bmm.cpp` | compute | ✓ `DataflowBuffer`, `matmul_init(dfb,dfb)`, `pack_tile(0, dfb)` |

  **Two CB-index free-function call sites were examined and neither is a holdover.** Recording the
  reasoning because both would false-fire on the shape cue alone:

  | Site | Call | Why not a Device 2.0 violation |
  |---|---|---|
  | `writer_unary_interleaved_start_id.cpp:19` | `get_local_cb_interface(dfb_id_out).fifo_page_size` | `get_local_cb_interface` is on the recipe's **sanctioned** list, and the list "does not turn on what object is in scope" — sanctioned even though `dfb_out` is in scope and exposes a getter |
  | `reader_bmm_8bank_output_tiles_partitioned.cpp:70,76` | `get_dataformat(dfb_id_in0)` | Not on the sanctioned list, but the result is consumed as `constexpr`, and **no wrapper-method replacement exists** for a constant expression (no `DataflowBuffer` can be `constexpr`). The holdover definition requires a replacement to exist; here it does not |

  Both are nonetheless **port-stage work** — see Port-work summary. This is the breadcrumb the
  recipe's Green bullet describes: a Metal 2.0 port moves the first onto the object, and keeps the
  second in free-function form with the binding token.

- **Feature compatibility:** every Appendix A entry scanned against the factory and its three
  kernels. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_config`, no `global_cb` parameter. This factory never reads `MatmulParams::global_cb` — the op's GCB use is confined to `matmul_multicore_reuse_mcast_1d`, which is a different factory and outside this audit |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | This factory creates **no semaphores at all** |

- **CB endpoints (GATE-free):** **all legal.** Three CBs, census run per node across all three
  kernels. Every one is a plain 1 producer + 1 consumer FIFO — no self-loop, no 1P+1C assignment
  needed, no multi-binding flag, no dead CB, no config-dependent flip (this factory has a single
  code path; the only branch is core-group count, which does not change any CB's endpoints).

  | CB | Producer (locked) | Consumer (locked) | Census | Disposition |
  |---|---|---|---|---|
  | `c_0` (in0) | reader — `reserve_back`/`push_back` | compute — `wait_front`/`pop_front` | 2 touchers, 1P+1C | plain 1:1, no action |
  | `c_1` (in1) | reader — `reserve_back`/`push_back` | compute — `wait_front`/`pop_front` | 2 touchers, 1P+1C | plain 1:1, no action |
  | `c_16` (out) | compute — `reserve_back`/`push_back` | writer — `wait_front`/`pop_front` | 2 touchers, 1P+1C | plain 1:1, no action |

  The reader's `dfb_in0.get_write_ptr()` at lines 71/77 (feeding `pad_last_ktile`) does **not** add a
  toucher: it is the CB's own FIFO producer peeking its own buffer, which a PRODUCER binding already
  covers.

- **Offset base pointers:** **GREEN.** The factory contains no `->address()` / `.address()` /
  `buffer()` expression at all. Runtime args carry the tensor objects themselves
  (`reader_desc.emplace_runtime_args(core, {a, b, ...})`, `writer_desc.emplace_runtime_args(core,
  {output, ...})`), so there is no host-side arithmetic that could fold an offset into a base. No
  Type 1, no Type 2, no Type 3, no Type 4. The offset-base-pointer triage doc lists no matmul row,
  and my own scan agrees with it.

- **TensorAccessor 3rd argument:** **N/A — no accessor in this factory passes a 3rd argument.** All
  three construction sites are 2-arg: `TensorAccessor(src0_args, src0_addr)` and
  `TensorAccessor(src1_args, src1_addr)` (reader lines 57-58), `TensorAccessor(dst_args, dst_addr)`
  (writer line 31). The subject never fires. This matches the 3rd-arg triage doc, whose `matmul`
  row is struck through as *"already fixed on main — every matmul TensorAccessor is now 2-arg."*

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — three, all **Case 1** (base address fed into a
  `TensorAccessor`, all memory access through the accessor):
  - `in0` — reader RTA slot 0 (`src0_addr`) → `TensorAccessor(src0_args, src0_addr)`.
  - `in1` — reader RTA slot 1 (`src1_addr`) → `TensorAccessor(src1_args, src1_addr)`.
  - `output` — writer RTA slot 0 (`dst_addr`) → `TensorAccessor(dst_args, dst_addr)`.

  All three arrive in the **tensor-object / `Buffer*`-binding form**, not `->address()`: the factory
  pushes the tensor into `emplace_runtime_args` and the framework auto-registers and patches it on
  cache hits. That shape is *correct today* — it is **not** the silent-wrong smuggled-pointer
  hazard (the sheet agrees: `Smuggled pointer = no`). It is still enumerated here because it is real
  port work: each becomes a `TensorParameter` + `TensorBinding`, and the address RTA plus its
  `TensorAccessorArgs` plumbing both disappear.

- **TensorParameter relaxation:** none.

- **TensorAccessor 3rd arg:** none — no site passes one.

- **CB endpoints:** all legal — three plain 1P+1C FIFOs, nothing to self-loop, assign, flag or drop.

- **Kernel-side metadata rewrites** (from the Device 2.0 breadcrumbs above):
  - `writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(dfb_id_out).fifo_page_size`
    → `dfb_out.get_entry_size()`. The value is declared `const uint32_t`, not `constexpr`, so the
    member getter fits (whitelist §B).
  - `reader_bmm_8bank_output_tiles_partitioned.cpp:70,76` — `get_dataformat(dfb_id_in0)` is declared
    `constexpr`, so it **keeps the free-function form** with the binding token:
    `get_dataformat(dfb::in0)`. Do not demote it to `const` to fit a member getter (whitelist §A).

- **`opt_level` — absent, and this is the failure mode with no diff to read.** `grep -n opt_level`
  on the factory returns **nothing**. A `KernelDescriptor` with no `opt_level` still resolves to the
  legacy per-kernel-type default: **`O3` for a `ComputeConfigDescriptor`**, `O2` for DM. Metal 2.0's
  `CompilerOptions` defaults to `O2` for both. So the port must set
  `compiler_options.opt_level = KernelBuildOptLevel::O3` **explicitly on both compute KernelSpecs**
  (core group 1 and core group 2). The two DM kernels need nothing.

- **Hardware config** — Style A (the factory resolves a TTNN `ComputeKernelConfig` via
  `get_compute_kernel_config_args`, factory line 55), so translate with
  `to_compute_hardware_config(device->arch(), config)`. **No dropped field:** all four
  helper-covered knobs (`math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`,
  `dst_full_sync_en`) are set on *both* compute descriptors (lines 216-219, 234-237).
  `packer_l1_acc` is resolved and explicitly discarded (`(void)packer_l1_acc;`, line 57) — it has no
  Metal 2.0 counterpart, so no action. Both DM kernels use plain `ReaderConfigDescriptor{}` /
  `WriterConfigDescriptor{}` (lines 147, 160) → the arch-agnostic
  `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
  `unpack_modes` needs no entry: no compute kernel here consumes a Float32 DFB.

- **Device-operation-class edits the port forces** — two of the three sanctioned exceptions apply:
  1. **Remove the pybound factory entry point.** `matmul_nanobind.cpp:1260-1274` is an entire
     `nb::class_<ttnn::prim::MatmulMultiCoreProgramFactory>` block whose only member is
     `create_descriptor`. That method vanishes at port time, so the whole block must be deleted.
     User-visible API surface change — record it in the port report under Handoff points.
     *(Leave the separate `nb::class_<MatmulDeviceOperation>` block at lines 1222-1237 alone: it
     binds device-op methods that survive the port, including the renamed hash hook.)*
  2. **Drop the pybind-hook-only factory parameter.** `create_descriptor` takes a fourth argument,
     `const std::optional<CoreRangeSet>& core_range_set`, which the factory body **ignores** — it is
     spelled `/*core_range_set*/` at `matmul_multicore_program_factory.cpp:31`. It exists solely so
     the pybind hook above can drive the factory. The fixed `create_program_artifacts` signature
     cannot carry it: drop the parameter (there is no production default to inline, since nothing
     reads it) and delete the hook that passed it. Same report handling as exception 1.

  Exception 3 does **not** apply — the op has a proper `program_factory_t` variant
  (`matmul_device_operation.hpp:24-31`), so this is not the direct-descriptor shape.

- **The op's cache key is not an edit.** There is no framework-visible custom hash to preserve or
  remove; the default reflection hash is in use. The renamed `compute_descriptor_program_hash`
  helper and its pybind name stay exactly as they are.

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader,
  no dual-instance work-split over a shared grid.

- **Cross-op / shared kernels: none — and the filename grep is a decoy here.** All three bound
  kernels are matmul-private with exactly one binder each. This deserves recording because the naive
  census badly misreports it:

  | Kernel | Raw filename hits | Real binders after disambiguation |
  |---|---|---|
  | `reader_bmm_8bank_output_tiles_partitioned.cpp` | 1 | 1 — this factory |
  | `bmm.cpp` | 3 | 1 — this factory. The other two are discardable: `moreh/sources.cmake:18` is a **build file** (and names a different file, `moreh_bmm/moreh_bmm.cpp`), and `moreh_matmul.cpp:5` is a **comment**, *"// Implemented based on bmm.cpp"* |
  | `writer_unary_interleaved_start_id.cpp` | **24 factories** | **1 — this factory** |

  The writer is the trap. Three same-named copies exist in the tree
  (`matmul/…`, `eltwise/unary/…`, `data_movement/slice/…`), and **every one of the other 23 binders
  binds a different copy** — 22 bind the `eltwise/unary` copy, 1 (`slice_program_factory_tile.cpp`)
  binds the `slice` copy. An exhaustive grep for the *bound path*
  `matmul/device/kernels/dataflow/writer_unary_interleaved_start_id` returns exactly one hit in the
  whole repo: `matmul_multicore_program_factory.cpp:155`. Those 23 are **same-named private copies**,
  which the recipe says to discard from the consumer list.

  **Rung-1 `_metal2` check, run locationally as specified:** `ls` of
  `matmul/device/kernels/dataflow/` shows **no** `_metal2` sibling. Two `_metal2` forks do exist in
  the tree — `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  and `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — but
  they are siblings of *other* copies, not of matmul's, so they do not count and must not be bound.

  **Consequence: the shared-kernel Caution does not apply. Convert all three kernels in place. Create
  no fork, add no pointer comment, touch no peer directory.**

- **Preserved multiplicity — two compute KernelSpecs, not one.** The factory calls
  `split_work_to_cores` and emits **two** `KernelDescriptor`s of `bmm.cpp`: `compute_desc_1` over
  `core_group_1` with CTA `num_output_tiles_per_core_group_1`, and `compute_desc_2` over
  `core_group_2` with `num_output_tiles_per_core_group_2` (the second is conditional on
  `core_group_2` being non-empty). The port must preserve this as **two KernelSpecs of the same
  source in two WorkUnitSpecs over disjoint node sets**, each binding the same three DFBs with the
  same roles. Because the node sets are disjoint each node still sees exactly one instance, so these
  are ordinary single-role bindings — **not** `allow_instance_multi_binding`, and not the
  same-grid two-toucher case. Demoting the per-group tile count to an RTA to collapse them into one
  KernelSpec is the documented anti-pattern.

- **RTA varargs:** none. Every kernel reads its arguments as distinct fields at constant indices in
  a block at the top — reader slots 0-11, writer slots 0-2. No loop-indexed reads, no
  data-selected index, no sentinel scan. All become named RTAs.

- **Positional CTAs to name:** `bmm.cpp` reads four positional CTAs
  (`get_compile_time_arg_val(0..3)` → `batch`, `Mt`, `Kt`, `Nt`); the reader reads two
  (`in0_last_ktile_w`, `in0_last_ktile_h`). All become named. The CB indices already arrive as
  *named* CTAs (`get_named_compile_time_arg_val("cb_in0")` etc.) — those become `DFBBinding`s, not
  named args.

- **Tests exist that pin this factory explicitly.** `tests/ttnn/unit_tests/gtests/test_matmul.cpp`
  carries cases commented as targeting `MatmulMultiCoreProgramFactory` — the *"fallback of last
  resort"* case at line 227, and a not-tile-aligned `[1,1,60,60] x [60,60]` case at line 276.
  Surfacing it because the porter must discover and confirm the test set themselves, and a factory
  reachable only as a fallback is easy to leave uncovered.

---

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. One function-call escape, no
  file-path escapes.

  | Op kernel | Donor file | Class | Functions called | Shape | Status |
  |---|---|---|---|---|---|
  | `reader_bmm_8bank_output_tiles_partitioned.cpp` | `ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp` | 4 — shared utility pool | `pad_last_ktile<DataFormat, uint32_t>(uint32_t)`, `pad_last_transposed_ktile<DataFormat, uint32_t>(uint32_t)` | raw `uint32_t` L1 address + `DataFormat` NTTP | ✓ |

  Neither function takes a `Semaphore`, a `TensorAccessor`, a `TensorAccessorArgs<N>`, a CB id, or a
  `CircularBuffer&` — so no `sem::` / `tensor::` handle is required at the boundary and the recipe's
  boundary assumption is not violated. The address argument is supplied by
  `dfb_in0.get_write_ptr()`, a public peek that crosses freely; the `DataFormat` NTTP is supplied by
  the `constexpr get_dataformat(...)` discussed above, which post-port becomes the binding-token
  form. **No donor-side change needed; no fork.** All other includes are `api/*`
  (LLK/HAL, donor class 1 — no concern).

  **Borrowed kernel files (file-path instantiation): none.** All three kernels are owned by matmul
  and instantiated only by matmul.

- **Relaxation candidates:** none. There is no custom hash to mine, and the sheet's
  `TensorParameter relaxation` is `none`.

- **TTNN factory analysis:** op-owned tensors — none. MeshWorkload need — none (`Execution Model =
  SPMD`, `Concept = descriptor`). Pybind `create_descriptor` — `matmul_nanobind.cpp:1260-1274`.
  Other risky pybind — the device-op class binding at `matmul_nanobind.cpp:1222-1237`, which
  exposes `create_output_tensors`, `compute_output_specs`, and `compute_program_hash` (mapped to
  the renamed helper); it survives the port untouched. Custom hash — none framework-visible.
  `get_dynamic_runtime_args` — absent. `override_runtime_arguments` — absent. Target concept —
  `ProgramSpecFactoryConcept`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **`in0_last_ktile_h` is hardcoded to 0, making a whole kernel branch dead.**
  `matmul_multicore_program_factory.cpp:135` sets `uint32_t last_ktile_h = 0;` unconditionally and
  passes it as the second CTA. The reader's entire
  `if constexpr (in0_last_ktile_h > 0) { … pad_last_transposed_ktile … }` block
  (`reader_bmm_8bank_output_tiles_partitioned.cpp:74-79`) is therefore unreachable from this
  factory. Sibling factories compute a real value for the transposed case, so this looks like a
  `transpose_a` path this factory does not implement rather than dead code per se — worth an owner's
  eye on whether the hardcoded 0 is a silent limitation or an intended one.

- **`create_descriptor`'s `core_range_set` parameter is accepted and ignored.** Spelled
  `/*core_range_set*/` at `matmul_multicore_program_factory.cpp:31`. The sibling
  `MatmulMultiCoreReuseOptimizedProgramFactory` genuinely uses its equivalent parameter; this one
  does not. A Python caller passing a `core_range_set` to `MatmulMultiCoreProgramFactory.create_descriptor`
  today gets it silently discarded. (The port removes the parameter, so this resolves itself.)

- **Dead preprocessor branches in matmul's private writer copy.** `OUT_SHARDED` and `BACKWARDS`
  (`writer_unary_interleaved_start_id.cpp:24,33`) are never defined by this factory — it emits no
  `defines` on the writer at all, and its output is always interleaved. Now that this copy has
  exactly one binder, both branches are unreachable. Not the port's to remove (it changes nothing
  functionally, and deleting them is out of scope), but a candidate cleanup for the op owner.

- **The sheet flags a perf note on this factory.** `Pointer patching perf issue? = "suspect perf
  regression (+ fixed latent bug)"` and `Formerly custom hashed? = "yes"`. Both refer to the earlier
  ProgramDescriptor migration, not to the Metal 2.0 port, and neither gates. Recorded so a
  post-port performance comparison is not misread: if this factory is measured slower than an old
  baseline, that regression may predate the Metal 2.0 port.

---

## Questions for the user

1. **Scope confirmation:** this audit covers `MatmulMultiCoreProgramFactory` only. The remaining
   seven factories in this directory each need their own audit — two of them (`…Mcast1D` and the
   `MeshWorkload…Mcast1D`) are already `no` on the sheet and will RED. Do you want those audited
   next, or should the port of this factory proceed first?

## Recipe notes

- **The shared-kernel census produced a 24-to-1 false positive, and the recipe's own guidance is
  what caught it.** `grep -rl writer_unary_interleaved_start_id.cpp` returns 24 factories; the
  correct answer is one. The recipe already says to grep the filename and then check each hit's
  bound *path* (and warns that a path grep is lossy because factories split the literal across two
  lines). Both halves of that were load-bearing here — the path is split across lines 154-155 in
  this very factory. Worth keeping the emphasis; this is the highest-leverage sentence in the
  Out-of-directory subject.

- **The rung-1 `_metal2` check being *locational* rather than a filename grep also mattered.** Two
  `_metal2` forks of this filename exist in the tree, and both are siblings of copies this factory
  does not bind. A tree-wide grep would have suggested reusing one.

- **Minor friction — the sanctioned-free-function list versus the Quasar kernel audit.** The same
  line, `get_local_cb_interface(dfb_id_out).fifo_page_size`, is *sanctioned* by this audit's
  Device 2.0 gate and a hard **GATE** in `cb_dfb_quasar_audit_helper.md`. Both are correct for their
  own arch and the Quasar doc's header warns you off it, but the two verdicts landing on one line of
  code is exactly the kind of thing that reads as a contradiction on a fast pass. The Green bullet's
  breadcrumb defuses it; I would not change the rule, only note that this factory is a live example
  if a worked case is ever wanted.
