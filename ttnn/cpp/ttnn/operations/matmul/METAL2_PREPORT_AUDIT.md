# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `SparseMatmulMultiCoreReuseMcast1DProgramFactory`,
on `SparseMatmulDeviceOperation`.**

This factory belongs to the **second** DeviceOperation in the `matmul` directory. Everything below —
concept, custom hash, runtime-arg hooks, pybind — was cross-checked against
`SparseMatmulDeviceOperation` and its own files under `device/sparse/`, **not** against the dense
`MatmulDeviceOperation`. The dense op's seven factories were not audited and nothing here is a
verdict on them.

- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - **`SparseMatmulMultiCoreReuseMcast1DProgramFactory`** ← **audited**
    (`device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.{hpp,cpp}`)
  - `SparseMatmulMeshWorkloadMultiCoreReuseMcast1DFactory` — declared and defined in the same files
    but **not** in the `program_factory_t` variant; see Misc anomalies
- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`) — seven factories, none audited

> **⚠ Sheet lookup hazard — two rows carry this exact DeviceOperation and Factory name.**
> They differ only in the `Op` column:
> - `Op = matmul` → the mainline factory, `Factory definition path` under
>   `ttnn/cpp/ttnn/operations/matmul/device/sparse/…`. **This is the audited row.**
> - `Op = experimental/quasar/matmul` → the quasar copy, path under
>   `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/…`. **Out of bounds for this audit and not
>   used for any finding.**
>
> Both happen to read `Is able to port? = no`, so the verdict is unaffected — but a reader grepping
> the sheet by factory name will get two hits, and the rows differ in `Model`
> (`resnet` vs `other`), `Uses llama kernels?` (`yes` vs `no`) and `TensorParameter relaxation`
> (`none` vs `n/a (quasar)`). Match on `Op` or on `Factory definition path`, not on the factory name.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` (sparse sub-tree) |
| **Overall** | **RED** — blocked on one gate |
| **DOps / Factories** | `SparseMatmulDeviceOperation` → `SparseMatmulMultiCoreReuseMcast1DProgramFactory` (its only variant alternative) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 4 bound kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Not assessed — informational subject, skipped (see Skipped subjects) |
| *Feature Support* — overall | **GREEN** — but see the GlobalCircularBuffer adjudication below |
| *Feature Support* — GlobalCircularBuffer | **N/A** — type present in the attribute struct and public API, **never consumed**; reasoning in full below |
| *Feature Support* — `address_offset` / GlobalSemaphore | N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No** — `Concept = legacy device-op` |
| *TTNN Readiness* — Concept (current) | `legacy device-op` — satisfies `ProgramFactoryConcept` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** — the declaration is **commented out** |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | Present, as part of the **legacy** concept signature (sheet correctly reads `n/a` for the PD-only column) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — neither the device-op class nor the factory is bound |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (per the sheet; reachable only after the PD migration) |
| *Port work* — Offset base pointer | **none** — 10 address args, every one a clean base |
| *Port work* — Tensor bindings (per binding) | Not assessed — informational subject, skipped |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — every construction is 2-arg |
| *Port work* — CB endpoints | Not assessed — informational subject, skipped |

---

## Result

**RED at factory level; no portable subset.** Blocked on a single gate:

- **TTNN factory concept** — the readiness sheet's `Is able to port?` reads **`no`**, attributable to
  `Concept = legacy device-op` (corroborated by `Op Classification = Legacy Op`). The op has not been
  migrated to the `ProgramDescriptor` API. Routed to the **TTNN / ProgramDescriptor-migration team**.

**This is the expected outcome for a legacy op, not an alarm.** The `ProgramDescriptor` migration is
a separate, ongoing workstream; the Metal 2.0 port becomes possible once this op's migration lands.
Note that `Diego validation = yes` and `Known op issues` is empty — the op is not flagged as broken,
merely unported.

Every other gate cleared: Device 2.0, Feature compatibility, Offset base pointers, and TensorAccessor
3rd argument. So when the PD migration lands, this factory has **no known second blocker**.

There is no portable subset: the concept gate is a property of the whole factory, not of one branch,
and the factory is the device-op's only variant alternative.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED.** The sheet's mainline row for
  (`matmul`, `SparseMatmulDeviceOperation`, `SparseMatmulMultiCoreReuseMcast1DProgramFactory`) reads
  **`no`**, with `Concept = legacy device-op`, `Op Classification = Legacy Op`,
  `Porting Target = ProgramSpecFactoryConcept`, `Model = resnet`.

  **The blocking column is `Concept`**, and the code confirms it precisely. The factory declares:

  ```
  hpp:14  struct shared_variables_t { … }
  hpp:24  using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
  hpp:26  static cached_program_t create(…);
  hpp:31  static void override_runtime_arguments(cached_program_t&, …);
  ```

  That is `ProgramFactoryConcept` — the **oldest** concept, the legacy `host_api.hpp` builder style.
  There is **no `create_descriptor`** anywhere, so the factory is not on
  `ProgramDescriptorFactoryConcept`, which the Metal 2.0 port requires as its starting shape. Routed
  to the **TTNN / PD-migration team**; the gate lifts when that migration lands, and the op is then
  re-audited against the rewritten code.

  **Lightweight cross-check — clean on every primary column**, verified against the *sparse*
  device-op:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `legacy device-op` | `create` + `override_runtime_arguments` + `cached_program_t`; no `create_descriptor` (hpp:24-36) | ✓ |
  | `Custom hash` | `no` | The `compute_program_hash` declaration is **commented out** — `sparse_matmul_device_operation.hpp:33` and its definition at `sparse_matmul_device_operation.cpp:504`. No live override, so the framework uses the default reflection hash | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` in the sparse tree | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across `device/sparse/` | ✓ |
  | `Override runtime args method?` | `n/a` | The method exists, but on a **legacy** device-op it is part of the legacy-concept signature, not the PD-only target-concept signal this column tracks. `n/a` is the correct value | ✓ |
  | `Pybind descriptor` | `no` | Neither `SparseMatmulDeviceOperation` nor the factory is bound. `matmul_nanobind.cpp` binds only the **user-facing `sparse_matmul` function** (`:1063`, `:1183`) | ✓ |
  | Factory-set match | 1 mainline row | `program_factory_t = std::variant<SparseMatmulMultiCoreReuseMcast1DProgramFactory>` — exactly one alternative (hpp:21) | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no`, and `Op-owned tensors?` is `no`.

  **Consequence worth recording for the eventual port:** because nothing is pybound here, the
  device-op-class edits that the port forces on the *dense* matmul factories — deleting a pybound
  `create_descriptor`, dropping a pybind-hook-only `core_range_set` parameter — **do not apply to
  this factory at all**. Its port will be confined to the factory body and its kernels.

- **Device 2.0 (every kernel used): GREEN.** All four kernels reachable from this factory are
  structurally Device 2.0 — `Noc` from `noc.h`, `DataflowBuffer` wrappers, `TensorAccessor`. Zero
  hits for broad Device-1.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`,
  raw `noc_async_read(` / `noc_async_write(`, raw `noc_semaphore_*`) and no non-sanctioned CB-index
  free-function holdovers. All four live in matmul's own `device/kernels/` tree (one copy each), so
  there is no donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Bound at | Device 2.0 evidence |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | `:468` | `Noc` ×13, `DataflowBuffer` ×3, `TensorAccessor` ×4 |
  | `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | `:486` | `Noc` ×2, `DataflowBuffer` ×1 |
  | `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `:500` | `Noc` ×23, `DataflowBuffer` ×5, `TensorAccessor` ×9 |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | `:555` | `DataflowBuffer` ×11, LLK compute APIs |

  This is worth stating plainly because it is the load-bearing good news: **the kernels are already
  Device 2.0**, so the PD migration is the only prerequisite standing between this op and a Metal 2.0
  port. The two prereq tracks do not have to be serialised.

- **Feature compatibility: GREEN.**

  | Feature | Status | Notes |
  |---|---|---|
  | **GlobalCircularBuffer** | **N/A** | Recognition signals fire on the *type*, but the feature is **not in use**. Full reasoning below — this one needed adjudication |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`. The factory uses two ordinary program semaphores |

  #### GlobalCircularBuffer — adjudicated N/A, with the reasoning stated

  **Two of Appendix A's "definitely this feature" bullets do fire on a literal reading:**
  - *"Any reference to the type `tt::tt_metal::experimental::GlobalCircularBuffer`"* —
    `sparse_matmul_device_operation_types.hpp:30` declares
    `std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;` as a field of
    `SparseMatmulParams`, and `:9` includes `global_circular_buffer.hpp`.
  - *"Op factory function signatures with parameter type
    `std::optional<const …GlobalCircularBuffer>&` (commonly named `global_cb`)"* —
    `sparse_matmul_device_operation.hpp:61` and `:79`.

  **But the feature is demonstrably not in use.** I traced every consumer:

  | Site | What it does with `global_cb` |
  |---|---|
  | `sparse_matmul_device_operation.hpp:61,79` | accepts it on the public API, defaulted `std::nullopt` |
  | `sparse_matmul_device_operation.cpp:535,585` | forwards it into `SparseMatmulParams` |
  | `sparse_matmul_device_operation.cpp:610,630` | forwards it again |
  | `…mcast_1d_optimized.cpp:43` | copies it into a `MatmulParams` **solely** so `get_program_config` can derive a program config |
  | *(nothing further)* | — |

  **No CB is ever created from it.** All six `CreateCircularBuffer` calls in the factory (`:579`,
  `:594`, `:629`, `:630`, `:648`, `:667`) are the **three-argument** form
  `(program, cores, cb_config)`. There is no `experimental::CreateCircularBuffer(…, global_cb)`
  four-argument overload, no `CircularBufferConfig::remote_index(...)`, no `remote_cb_config`, and no
  `.global_circular_buffer` field on any `CBDescriptor` anywhere under `device/sparse/`.

  So the disambiguator Appendix A actually names — *"the `CreateCircularBuffer(..., global_cb)` /
  `.remote_index(` construction"* — is **absent**. The parameter is accepted, threaded through four
  layers, and dropped.

  **Verdict: N/A, not RED.** This is a determinable case rather than an ambiguous one, so the
  conservative default does not apply — and calling it RED would misroute work to the Metal 2.0 team
  when the actual blocker is the ProgramDescriptor migration. Recorded here in full so the judgement
  is auditable, and raised as a question below, because if the parameter is *intended* to be wired up
  later the answer changes.

- **Offset base pointers: GREEN.** Ten address-valued runtime args exist — `:704` (in0), `:715`
  (sparsity), `:737` (in1), `:746` (in1 sparsity), `:750` (out) in the builder, and `:838`, `:839`,
  `:849`, `:850`, `:851` in `override_runtime_arguments` — and **every one is a bare `->address()`
  with no host-side arithmetic**. A scan for `address() +`, `addr + …` and `+ …offset…` returns zero
  hits. No Type 1, no Type 2, no Type 3, no Type 4.

  These raw addresses in runtime args are the **legacy concept's sanctioned mechanism**, not the
  smuggled-pointer hazard: on `ProgramFactoryConcept` the framework genuinely calls
  `override_runtime_arguments` on every cache hit, and `:838-851` is exactly that method re-patching
  them. The sheet's `Smuggled pointer = no` is consistent with this reading.

- **TensorAccessor 3rd argument: N/A.** No accessor in any of the four kernels passes a third
  (page-size) argument — every construction is 2-arg. The subject never fires.

---

## Skipped subjects  *(disclosure — these were not run)*

This is the recipe's explicitly-named **"RED short-circuit: the op is still on the legacy imperative
API"** case, which directs: run every gate-bearing subject in full, skip the seven purely-informational
ones, and record each skip.

**Which side does the RED clear on? Op-code side — unambiguously.** A `ProgramDescriptor` migration
rewrites the factory body wholesale: `create` becomes `create_descriptor`, the imperative
`CreateCircularBuffer` / `CreateKernel` / `CreateSemaphore` calls become descriptors, the runtime-arg
construction changes shape, and `override_runtime_arguments` is restructured or removed. That is
precisely the code the seven subjects would describe, so producing that detail now would be unread
and stale before a port could use it.

| Subject | Note |
|---|---|
| TTNN porting shape | skipped — whole-factory RED, no portable subset; re-audit on unblock. The sheet's `Porting Target = ProgramSpecFactoryConcept` is recorded, but re-derive it after the migration |
| TensorParameter relaxations | skipped — the sheet's cell reads `none`, the value that clears; no further analysis run |
| TensorParameter analysis | skipped — per-binding Case 1 / Case 2 census not run; re-audit on unblock |
| CB endpoints | skipped — per-`(CB, config)`-per-node census not run; re-audit on unblock |
| Out-of-directory coupling | skipped — the donor-shape inventory was not run. Nothing gate-relevant is lost: the Device 2.0 gate above already covers every kernel this factory uses, and all four are matmul-owned with a single copy each, so there is no donor blocker to name |
| RTA varargs | skipped — re-audit on unblock |
| Incidental anomalies | not swept — opportunistic by design; what surfaced while running the gates is below |

**None of these is a clean result.** They are deferred, not owed.

---

## Misc anomalies  *(team-only, non-gating; noticed while running the gates, not a sweep)*

- **A public API parameter is accepted and silently ignored.** `sparse_matmul(...)` takes
  `const std::optional<const GlobalCircularBuffer>& global_cb` (`sparse_matmul_device_operation.hpp:61`,
  `:79`), threads it through `SparseMatmulParams` into the factory, and the factory never uses it for
  anything (see the Appendix A adjudication). A caller passing a GlobalCircularBuffer today gets it
  discarded with no diagnostic. Because `SparseMatmulParams` feeds the default reflection hash, the
  field also participates in the program-cache key, so two otherwise-identical calls differing only
  in `global_cb` will miss the cache while producing identical programs. Worth an owner's decision:
  wire it up, or remove it from the signature.

- **A second factory struct is defined but never dispatched.**
  `SparseMatmulMeshWorkloadMultiCoreReuseMcast1DFactory` is declared at
  `…mcast_1d_optimized.hpp:38` and fully defined at `…mcast_1d_optimized.cpp:859-879+`
  (`create_mesh_workload` and `override_runtime_arguments`), but
  `SparseMatmulDeviceOperation::program_factory_t` is
  `std::variant<SparseMatmulMultiCoreReuseMcast1DProgramFactory>` — a single alternative — and no
  other file in the tree references the MeshWorkload struct. It is compiled dead code. This is also
  why the sheet correctly carries one mainline row rather than two; flagged so a future auditor does
  not read the second struct as a missing row.

- **The custom program hash is commented out rather than absent.**
  `sparse_matmul_device_operation.hpp:33` and `.cpp:504` both carry a commented-out
  `compute_program_hash` declaration/definition. Functionally equivalent to having none — the
  framework uses the default reflection hash — but a grep for `compute_program_hash` hits it, so a
  future cross-check should read the line rather than count the match.

- **The device-op declares `validate_on_program_cache_hit`** (`sparse_matmul_device_operation.hpp:24`)
  in addition to `validate_on_program_cache_miss`. Uncommon among the matmul factories; noted because
  a PD migration will need to decide where that validation lands.

---

## Questions for the user

1. **Is `global_cb` on `sparse_matmul` intended to be wired up, or is it vestigial?** The parameter
   is accepted on the public API and threaded to the factory, which ignores it; no CB is created from
   it. I have adjudicated Appendix A as **N/A** on that basis rather than REDing a second gate,
   because calling it in-use would misroute work to the Metal 2.0 team when the real blocker is the
   PD migration. **If the intent is to attach a GlobalCircularBuffer later, that changes the answer**
   — Metal 2.0 has no `GlobalDataflowBuffer`, so the feature would then be a genuine second blocker
   and would need its own entry against the Metal 2.0 track.

2. **Re-audit trigger.** The only blocker is the ProgramDescriptor migration, and the kernels are
   already Device 2.0 — so nothing else needs to happen in parallel. Should this factory be
   re-audited automatically when that migration lands?

3. **Scope.** This completes the sparse device-op. Seven factories on `MatmulDeviceOperation` remain;
   which, if any, should follow?

## Recipe notes

- **The legacy-API short-circuit is well specified and I used it verbatim.** Being told explicitly to
  run all gate-bearing subjects but skip the seven informational ones — and *why* (the migration
  rewrites the code they describe) — removed what would otherwise have been a judgement call, and the
  resulting report says the useful thing: the concept gate is the only blocker, and the kernels are
  already clear, so the two prereq tracks need not be serialised.

- **Appendix A's GlobalCircularBuffer recognition over-fires on a merely-declared field.** Two
  "definitely this feature" bullets — the type reference and the `std::optional<const
  GlobalCircularBuffer>&` factory-signature bullet — match an op that accepts the parameter and never
  uses it. The false-positive guard anticipates the *scalar* near-miss (`num_global_cb_receivers`) but
  not the *typed-but-unconsumed* one. The Action bullet's own disambiguator (the
  `CreateCircularBuffer(..., global_cb)` / `.remote_index(` construction) is what actually resolved
  it, so the material is all there — but a guard bullet saying "a declared-but-never-constructed
  `global_cb` parameter is not in use; key on the construction" would make the call mechanical
  instead of a judgement I had to reason out and defend.

- **The sheet's one-row-per-(op, DeviceOperation, factory) key needs the `Op` column to disambiguate
  quasar copies.** Grepping by factory name returned two rows with identical DeviceOperation and
  Factory values, differing only in `Op` and `Factory definition path`. The readiness doc says to
  look up by op path, which does resolve it — but the audit recipe's cross-check instructions talk in
  terms of "your op's row(s)", and for any op with a quasar twin that phrasing is one grep away from
  reading the wrong row. A sentence warning that `experimental/quasar/*` rows shadow mainline ones
  would be cheap insurance, especially since the quasar tree is out of bounds everywhere else.
