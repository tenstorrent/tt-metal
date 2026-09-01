# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`.**
The op directory holds two DeviceOperations and eight ProgramFactories; this audit covers a single
factory by request. The other seven are named below for disambiguation only and were **not**
audited — no statement in this report should be read as a verdict on any of them.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`** ← **audited**
    (declared `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp:42`)
  - `MatmulMultiCoreProgramFactory` — not audited
  - `MatmulMultiCoreReuseOptimizedProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast1DProgramFactory` — not audited *(shares the same `.cpp` as the
    audited factory; see the scoping note below)*
  - `MatmulMultiCoreReuseMcast2DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` — not audited
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

> **Scoping note — two factories share one implementation file.**
> `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (5,885 lines) hosts both the
> audited factory and `MatmulMultiCoreReuseMcast1DProgramFactory`. They do not share code paths:
> the audited factory builds through `matmul_multi_core_reuse_mcast_1d_optimized_` (line 5152) into
> the imperative `process_*_program_and_create_override_variables` functions, while the sibling
> builds through `create_program_mcast_in0_descriptor` (3141) / `create_program_mcast_in1_descriptor`
> (4217). Every line-referenced finding below was checked against those function boundaries and is
> in the audited factory's reachable code.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **RED** — blocked on two independent gates |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 in-scope kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Not assessed — informational subject, skipped (see Skipped subjects) |
| *Feature Support* — overall | **RED** |
| *Feature Support* — GlobalCircularBuffer | **RED** — in use, definitive signals at 4 sites |
| *Feature Support* — CBDescriptor `address_offset` (non-zero) | N/A |
| *Feature Support* — GlobalSemaphore | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No** — attributable to two blocking columns |
| *TTNN Readiness* — Concept (current) | `legacy (MeshWorkload)` |
| *TTNN Readiness* — Secretly SPMD | `yes` — sheet records the reason as `GlobalCircularBuffer` |
| *TTNN Readiness* — Custom hash | No (not a gate) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | `n/a` on the sheet (PD-only column); the factory does declare one, as part of its legacy-concept signature |
| *TTNN Readiness* — Pybind `create_descriptor` | `PR` — device-op-level; this factory has no pybind of its own |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **None yet** — see TTNN porting shape note |
| *Port work* — Offset base pointer | **none** — every address RTA carries a clean base |
| *Port work* — Tensor bindings (per binding) | Not assessed — informational subject, skipped |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — no accessor in any in-scope kernel passes a 3rd argument |
| *Port work* — CB endpoints | Not assessed — informational subject, skipped |

---

## Result

**RED at factory level; no portable subset.** Blocked on **two independent gates**, which clear on
different sides and can be worked in parallel:

1. **TTNN factory concept** — the readiness sheet's `Is able to port?` reads `no`, attributable to
   `Concept = legacy (MeshWorkload)` and `Smuggled pointer = yes` (`Op Classification = Broken Op`).
   Routed to **TTNN**: the op must be fixed and migrated before a Metal 2.0 port is possible.
2. **Feature compatibility — GlobalCircularBuffer** (Appendix A, UNSUPPORTED). Routed to
   **Metal 2.0**: unblocks when the user-managed `GlobalDataflowBuffer` lands.

**Why no portable subset exists.** The factory has two reachable build paths (`gather_in0` and
`mcast_in0`, see Gate detail), and one of them — `gather_in0` without a `global_cb` — does avoid the
GlobalCircularBuffer. But that does not yield a subset, because the **TTNN factory-concept gate is
structural and unconditional for this factory**: it satisfies `MeshWorkloadFactoryConcept`
(`cached_mesh_workload_t` + `create_mesh_workload` + `override_runtime_arguments`,
`…mcast_1d_program_factory.hpp:42-57`), which is neither of the two Metal 2.0 target concepts, and
no code path inside it escapes that. The smuggled-pointer finding is likewise factory-wide. So the
blocking shape is structural rather than one branch among siblings.

Neither gate is a permanent blocker, and neither is the porter's to clear.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`) reads
  **`no`**. This is the common, fully-attributable case — two blocking columns explain it:

  | Blocking column | Value (verbatim) | What it means | Routed to |
  |---|---|---|---|
  | `Concept` | `legacy (MeshWorkload)` | Not on the `ProgramDescriptor` API. This is the **expected** outcome for a legacy factory, not an alarm | **TTNN / PD-migration team** — unblocks when this factory's migration lands |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `yes` | The op reads as *broken*, not merely unported — corroborated by `Op Classification = Broken Op` and `Diego validation = no` | **TTNN**, who fix the op first |

  Supporting cells, recorded but not themselves blocking: `Porting Target = (N/A)`,
  `Execution Model = SPMD (+ per-device args)`, `Secretly SPMD Workload? = yes` with
  `Why secretly SPMD? = GlobalCircularBuffer`, `Op-owned tensors? = no`,
  `TensorParameter relaxation = none`, `Custom hash = no`, `Backdoor custom hash = no`,
  `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = n/a`,
  `Pybind descriptor = PR`, `Model = both`, `Uses llama kernels? = yes`.

  **Lightweight cross-check — clean on every primary column.** The sheet and the code agree; there
  is no spreadsheet-broken finding here.

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `legacy (MeshWorkload)` | `cached_mesh_workload_t` (hpp:44), `create_mesh_workload` (hpp:46), `override_runtime_arguments` (hpp:52) — the legacy mesh-workload signature. No `create_descriptor` on this struct | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op. A `compute_descriptor_program_hash` helper exists at `matmul_device_operation.hpp:50`, carrying a comment that it is *deliberately* not named `compute_program_hash` so the framework does **not** detect a custom hash; it is reached only through a pybind alias. The framework therefore uses the default reflection hash | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across `ttnn/cpp/ttnn/operations/matmul/` | ✓ |
  | `Override runtime args method?` | `n/a` | The factory *does* declare `override_runtime_arguments` (hpp:52), but on a legacy-concept factory that is part of the legacy signature, not the PD-only target-concept signal this column tracks. `n/a` is the correct value | ✓ |
  | `Pybind descriptor` | `PR` | This factory is **not** pybound (zero hits for its name in `matmul_nanobind.cpp`). The column tracks a `nb::class_` of the *device op*, which is a device-op-level fact shared across all eight rows — not a per-factory mismatch | ✓ |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1 with the sheet's 8 rows — no phantom row, no missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` (it is only ever `yes` on
  `descriptor` / `WorkloadDescriptor`), and `Op-owned tensors?` is `no`.

- **Device 2.0 (every kernel used): GREEN.** All eight kernels reachable from this factory are
  structurally Device 2.0 — `Noc` from `noc.h`, `DataflowBuffer` wrappers, `TensorAccessor`. No
  broad Device-1.0 idioms anywhere: zero hits for `InterleavedAddrGen`, `ShardedAddrGen`,
  `InterleavedPow2AddrGen*`, raw `noc_async_read(` / `noc_async_write(`, or raw
  `noc_semaphore_wait` / `_set` / `_inc`. No isolated CB-index free-function holdovers either
  (no bare `get_read_ptr(cb_id)` / `get_write_ptr(cb_id)` with a wrapper in scope).

  All eight live in matmul's own `device/kernels/` tree, so there is no donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Build path | Device 2.0 |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | mcast_in0 | ✓ |
  | `dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | mcast_in0 (in0 sharded) | ✓ |
  | `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | mcast_in0 | ✓ |
  | `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | mcast_in0 | ✓ |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | mcast_in0 | ✓ |
  | `dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp` | gather_in0 | ✓ |
  | `dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | gather_in0 | ✓ |
  | `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | gather_in0 | ✓ |

  **Sanctioned free functions are present and are not flagged.** Seven of the eight kernels call
  `get_local_cb_interface(...)` and/or `get_tile_size(...)` — both are on the recipe's sanctioned
  list, which "does not turn on what object is in scope," so they stay sanctioned even where a
  `DataflowBuffer` is in scope and exposes an equivalent getter. These are Metal 2.0 *port-stage*
  rewrites (onto the object, or kept in free-function form with the binding token where the value is
  `constexpr`), not Device 2.0 violations. Recorded because the shape cue false-fires here.

- **Feature compatibility: RED — GlobalCircularBuffer in use.**

  | Feature | Status | Notes |
  |---|---|---|
  | **GlobalCircularBuffer** | **RED** | Four definitive signals, all in this factory's reachable code — see detail below |
  | CBDescriptor `address_offset` (non-zero) | N/A | Eleven `UpdateDynamicCircularBufferAddress` calls exist (2985-3103), but **all are the three-argument `(program, cb_handle, tensor)` form**, which Appendix A's false-positive guard explicitly excludes. No four-argument offset overload, no `.address_offset` field, no `set_address_offset`, no `cb_descriptor_from_sharded_tensor`. These are the ordinary borrowed-memory (dynamic CB) pattern, which is a mechanical porting-recipe translation via `borrowed_from` and not an Appendix A entry |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include |

  #### GlobalCircularBuffer — UNSUPPORTED (RED)

  **Recognition signals that fired** (Appendix A lists each of these as definitive on its own; all
  four are present):

  | Signal | Sites |
  |---|---|
  | Type `tt::tt_metal::experimental::GlobalCircularBuffer` in a factory signature | `…mcast_1d_program_factory.cpp:104` (mcast_in0 build), `:2159` (gather_in0 build), `:2999`, `:3076`, `:3121` (override path), `:5178` (the factory's own build entry point) |
  | **Construction-by-consumption** — the 4-arg `experimental::CreateCircularBuffer(program, cores, cfg, *global_cb)` | `:849` (mcast_in0), `:2326` (gather_in0) |
  | `CircularBufferConfig::remote_index(...)` | `:841`, `:2322` |
  | `remote_cb_config` identifier (the "remote CB" idiom) | `:840`, `:844`, `:2320`, `:2325` |

  **This is not an incidental use — it is half the reason the factory exists.** Factory selection at
  `matmul_device_operation.cpp:2204` routes into this factory precisely when
  `gather_in0 || global_cb.has_value()`, with an in-code comment stating the reason: *"ProgramDescriptor
  cannot attach an experimental GlobalCircularBuffer."* So the GCB is intrinsic to the factory's
  purpose rather than confined to a branch of it.

  **Expected resolution:** not yet supported in Metal 2.0. A GlobalCircularBuffer is a *user-managed*
  buffer whose eventual analog is the (unimplemented) user-managed `GlobalDataflowBuffer` — the
  mapping is by **lifetime**. It has no DataflowBuffer destination today: not the local
  `DataflowBuffer`, and **not** the `CrossNodeDataflowBuffer` stub, despite the legacy *"remote CB"*
  nickname (that stub is a separate *ephemeral* construct split on the locality axis, with no legacy
  analog). The port becomes possible once `GlobalDataflowBuffer` support lands on `KernelSpec` /
  `DataflowBufferSpec`.

  **Confinement, for routing.** Within the shared `.cpp`, every GCB site sits in the audited
  factory's reachable code — the `process_mcast_in0_*` (77-1191), `process_gather_in0_*`
  (2124-2917) and `override_*_program_parameters` (2918-3140) functions. The sibling factory's
  descriptor builders (3141-5150) contain **no** GCB reference, consistent with the selection
  comment above. Appendix A's own guidance anticipates this shape for matmul and directs a
  factory-granularity RED rather than an op-wide one — which is what this report records. Whether
  the op's other factories are clean is **not** a claim this audit makes; each needs its own audit.

- **Offset base pointers: GREEN.** Every address-valued runtime arg in this factory passes a
  **clean base**. A scan for a host-side fold (`address() +`, `addr + …`, `+ …offset…`) across the
  whole file returns **zero** hits. The address RTAs in the reachable paths — `:1075` (in0),
  `:1114` (in1), `:1127` (out), `:1162` (bias) on the mcast_in0 build; `:2785` (in1) on the
  gather_in0 build; and their counterparts patched by index in the override helpers (`:2956-2992`,
  `:3036-3069`, `:3090-3115`) — are all bare `tensor.address()`. No Type 1 (raw offset arg), no
  Type 2 (accessor-fed offset arg), no Type 3 (`address_offset`), no Type 4 (`narrow`). My own scan
  is the source of truth here and it agrees with the checked-in triage, which lists no matmul row.

  Note for the reader: this is a *different question* from the sheet's `Smuggled pointer = yes`.
  Raw buffer addresses in RTAs unquestionably exist here — that is the TTNN-gate finding above. What
  this gate asks is whether any of them has an offset folded into the base, and none does.

- **TensorAccessor 3rd argument: N/A.** No accessor in any of the eight in-scope kernels passes a
  third (page-size) argument — every construction is 2-arg. The subject never fires, so there is
  nothing to classify and nothing to drop.

---

## Skipped subjects  *(disclosure — these were not run)*

This is a whole-factory RED with **no portable subset**, so per the recipe's Red-outcome scoping
rule the seven purely-informational subjects are deferred to the re-audit. **The judgement call:
which side does the RED clear on?** It is mixed, and I judged it **op-code side**:

- `Concept = legacy (MeshWorkload)` clears via a TTNN/PD migration that **rewrites this factory's
  body** — the imperative per-coord `Program` construction, its CB creation, and its RTA layout.
- `Smuggled pointer = yes` clears via a TTNN fix that **restructures how addresses reach the
  kernels** — precisely what a tensor-binding census would describe.
- Only the GlobalCircularBuffer blocker clears with the op untouched (framework side).

Two of the three blockers therefore rewrite the very code the seven subjects would document, so
producing that detail now would be unread and stale by the time a port is possible. A full
CB-endpoint census here would be the expensive one — roughly fifteen CBs across two build paths and
several sharding configurations, each needing a per-`(CB, config)`-per-node verdict.

Each skipped subject, with its one-line note — **none of these is a clean result**:

| Subject | Note |
|---|---|
| TTNN porting shape | skipped — whole-factory RED, no portable subset; re-audit on unblock. Partial finding recorded below |
| TensorParameter relaxations | skipped — the sheet's cell reads `none`, which is the value that clears; no further analysis run |
| TensorParameter analysis | skipped — per-binding Case 1 / Case 2 census not run; re-audit on unblock |
| CB endpoints | skipped — per-`(CB, config)`-per-node census not run; re-audit on unblock |
| Out-of-directory coupling | skipped — the donor-shape inventory was not run. Nothing gate-relevant is lost: the Device 2.0 gate above already covers every kernel this factory uses, and all eight are matmul-owned, so there is no donor blocker to name |
| RTA varargs | skipped — re-audit on unblock |
| Incidental anomalies | not swept — this subject is opportunistic by design; what I happened to notice while running the gates is recorded below |

**Partial finding — TTNN porting shape (recorded because the lookup was cheap and the answer is
load-bearing for planning).** The sheet marks this factory `Secretly SPMD Workload? = yes` with
`Why secretly SPMD? = GlobalCircularBuffer` — that is, it wears the mesh-workload shape to get a
capability, not because its per-coord programs genuinely differ. That is the resource-workaround
shape, which normally collapses onto a single-program concept. But the collapse is not available
here: the reason is a GlobalCircularBuffer, and Metal 2.0 has no construct to carry one. So the
target concept is recorded as **none yet** — not because the op is genuinely multi-program, but
because the feature its workload shape exists to obtain is unimplemented. Re-derive this at
re-audit; it may change entirely once `GlobalDataflowBuffer` lands.

---

## Misc anomalies  *(team-only, non-gating; noticed while running the gates, not a sweep)*

- **One of the file's three build paths is unreachable from this factory.**
  `process_mcast_in1_program_and_create_override_variables` (lines 1192-2123) cannot be reached
  through `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`. Selection requires
  `gather_in0 || global_cb.has_value()` (`matmul_device_operation.cpp:2204`); `gather_in0` is
  dispatched first inside `matmul_multi_core_reuse_mcast_1d_optimized_`, and for the remaining case
  the validator at `matmul_device_operation.cpp:1823-1826` fires
  *"global_cb without gather_in0 is supported only for mcast_in0=true"*. So `mcast_in0` is forced
  and `mcast_in1` is never selected. The path is live only through
  `matmul_multi_core_reuse_mcast_1d_optimized_helper`, which external CCL fused ops call directly.
  Not a defect — but it means a reader tracing this factory will find a third build path that its
  own dispatch can never take, and I scoped the Device 2.0 kernel set accordingly.

- **`smuggled-rta-ok` annotations.** Five sites in this file carry a trailing
  `// smuggled-rta-ok` comment (e.g. `:1162`, `:2034`), part of a repo-wide convention marking
  raw-address RTAs that have been deliberately reviewed — the same marker appears in conv2d, move,
  all_to_all_dispatch and ring_attention. Recorded because the sheet independently flags this
  factory `Smuggled pointer = yes`, and a reader reconciling the two should know the annotations
  exist and are not an assertion that the pointer is fine under Metal 2.0.

---

## Questions for the user

1. **Re-audit trigger.** Both blockers are owned elsewhere (TTNN for the concept + smuggled pointer;
   Metal 2.0 for `GlobalDataflowBuffer`). Do you want this factory re-audited when the *first* of
   them clears, or only once both have?

2. **Scope.** Six factories in this directory remain unaudited, plus the sparse device-op's factory.
   Should those follow, and in what order?

## Recipe notes

- **The `UpdateDynamicCircularBufferAddress` false-positive guard earned its place.** A scan for
  `address_offset` signals returns eleven hits in this file, and every one is the benign
  three-argument form. Without the guard's explicit *"three-argument form with no offset → not this
  rule"* line, this would have read as a second Appendix A RED. Worth keeping exactly as worded.

- **Friction — the `Override runtime args method?` column needed the collision note to resolve.**
  This factory declares an `override_runtime_arguments`, and the sheet says `n/a`. Those look like a
  conflict until you apply the recipe's note that on a *legacy* device-op the same method name is
  part of the legacy-concept signature rather than the PD-only target-concept signal. The note is
  present and correct; flagging only that this factory is a clean worked example of it, since the
  method is right there in the header next to the concept the column is asking about.

- **Suggestion: the Red-outcome scoping rule could name the mixed case.** The rule asks which side a
  RED clears on and gives op-code-side and elsewhere-side examples, but this factory is blocked on
  *both* kinds at once (a TTNN op fix and an unimplemented Metal 2.0 feature). I resolved it by
  reasoning that any op-code-side blocker is sufficient to make the informational detail stale, so
  the mixed case skips. That reading seems right but the doc does not say it, and a different
  auditor could plausibly run all seven subjects on the same evidence.
