# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/debug` (`ApplyDeviceDelayDeviceOperation`)

- **`ApplyDeviceDelayDeviceOperation`** (single-descriptor, direct-descriptor op — no `program_factory_t` wrapper)
  - `create_descriptor(...)` in `device/apply_device_delay_device_operation.cpp` — **per-mesh-coordinate variant** (takes `const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate`)
  - Kernel: `device/kernels/dataflow/device_delay_spin.cpp` (single-core spin kernel)
  - Donor: `spin()` from `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/debug` |
| **Overall** | **RED** — framework-blocked (wait-for-feature); **no portable subset** |
| **DOps / Factories** | `ApplyDeviceDelayDeviceOperation` → single direct `create_descriptor` (per-coord variant) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — kernel performs no data movement; nothing to migrate |
| *Prereqs* — Cross-op escapes | Ok — one in-family/shared donor (`data_movement/common/kernels/common.hpp::spin`), Device-2.0 clean |
| *Feature Support* — overall | GREEN (no Appendix A entry fires) — see caveat below |
| *Feature Support* — Variadic-CTA | Ok — one fixed CTA at a constexpr index |
| *TTNN Readiness* — `Is able to port?` (the sheet's gate cell) | **`yes`** (sheet) — **but the derived verdict does not model this op's blocker; see Result** |
| *TTNN Readiness* — Concept (current) | `descriptor` (per-coord `mesh_dispatch_coordinate` variant) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (concept is `descriptor`, not `WorkloadDescriptor`) — but **genuinely per-coord multi-program in behavior** |
| *TTNN Readiness* — Is safe to port? | `yes` (sheet) |
| *TTNN Readiness* — Custom hash | No (cross-check confirms: no `compute_program_hash`) |
| *TTNN Readiness* — Runtime-args update | No (cross-check confirms: no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (cross-check confirms: nanobind binds the host fn `apply_device_delay`, not `create_descriptor`) |
| *TTNN Readiness* — Op-owned tensors | No (blank on sheet; `tensor_args_t {}` is empty) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` — **structurally cannot carry this op (see Result)** |
| *Port work* — Offset base pointer | none — op has **no tensors / no address RTAs** |
| *Port work* — Tensor bindings (per binding) | none — op has **no tensors** |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no `TensorAccessor` (no tensors) |
| *Port work* — CB endpoints | none — op has **no CBs** |

## Result

**RED at op level; no portable subset.** The port is **framework-blocked** (wait-for-feature), routed to the **Metal 2.0 framework team**, with a reconciliation note to the **readiness-sheet owner** (the sheet's `Is able to port? = yes` does not model this op's blocker).

Two independent, structural walls block the transition of this `descriptor` op to its only Metal 2.0 target, `MetalV2FactoryConcept` (the `create_program_artifacts` → `ProgramSpec` path):

1. **Per-mesh-coordinate program divergence is not expressible in `MetalV2FactoryConcept`.** This op's entire purpose is to bake a *different* compile-time arg into the kernel on *each* device — `kernel_desc.compile_time_args = {operation_attributes.delays[mesh_coordinate[0]][mesh_coordinate[1]]}` (`device/apply_device_delay_device_operation.cpp:80`), reached through the per-coord `create_descriptor(..., mesh_dispatch_coordinate)` variant (`.hpp:46-50`, `.cpp:59-89`). The descriptor adapter honors this by calling the factory once **per coordinate** (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:596-600`, `create_descriptor_uses_mesh_dispatch_coordinate()`), so every device genuinely gets a distinct program. The Metal 2.0 target has **no per-coordinate hook**: `MetalV2MeshWorkloadFactoryAdapter::create_mesh_workload` calls `create_program_artifacts(attrs, tensor_args, tensor_return_value)` **once** — no `mesh_dispatch_coordinate` parameter exists — and stamps that **single** `ProgramSpec` identically across **every** coordinate range (`mesh_device_operation_adapter.hpp:887, 907-913`). Porting would collapse all per-device delays to one value: silent wrong behavior, with nothing to flag it. This is the same class of framework wall documented for genuine per-coord ops (`MetalV2FactoryConcept` and the mesh-workload path being mutually exclusive; `operation_concepts.hpp:91-92`) — a per-coord op cannot ride the ProgramSpec codegen path today.

2. **Tensorless op — `MetalV2FactoryConcept` adapter cannot run it.** `tensor_args_t` is empty (`.hpp:31`) and the op returns empty tensor vectors (`.cpp:51,56`). The MetalV2 adapter sources the `MeshDevice` from the first tensor in `tensor_args` and `TT_FATAL`s if there is none: *"MetalV2 factory adapter requires at least one Tensor in tensor_args to source the MeshDevice"* (`mesh_device_operation_adapter.hpp:876-881`). A tensorless op therefore cannot be dispatched under `MetalV2FactoryConcept` at all — an independent blocker even if (1) were resolved.

Neither wall is a single conditional branch — both are unconditional properties of the op (its reason to exist is per-device divergence; it inherently owns no tensors). So there is **no clean factory subset** to carve out. The op remains fully functional on the current `descriptor` per-coord path; it simply cannot advance to the Metal 2.0 concept until the framework grows a per-coordinate (and tensorless) `create_program_artifacts` path.

**Future path:** unblocks when Metal 2.0 gains a ProgramSpec-backed per-mesh-coordinate dispatch path (a `create_program_artifacts` variant that receives a `MeshCoordinate` and can vary the spec/CTAs per coord) **and** a tensorless MeshDevice source for that adapter. Both are framework-team features, not op-side or Device-2.0 work.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** The readiness sheet (row 126, fetched fresh this run) reads **`Is able to port? = yes`**, and the cheaply-checkable cross-check columns all **match the code** — `Concept = descriptor` (direct `create_descriptor`, no `program_factory_t`), `Custom hash = no` (no `compute_program_hash`), `Runtime-args update = no` (no `get_dynamic_runtime_args`/`override_runtime_arguments`), `Pybind descriptor = no` (nanobind binds the host fn `apply_device_delay`, not `create_descriptor`), `Op-owned tensors? = blank`, `Secretly SPMD = blank`. So the sheet's *shape* conjuncts and correctness axis are internally consistent and verified. **The gate nonetheless REDs**, because the sheet's `Is able to port?` derivation formula does not include a conjunct for per-mesh-coordinate program divergence or tensorless dispatch — the two framework walls in Result. **Route:** Metal 2.0 framework team (the feature gap); **and** the readiness-sheet owner to reconcile the `yes` (this op is a genuine per-coord multi-program op that the current ProgramSpec path cannot carry — see Recipe notes).
- **Device 2.0 (every kernel used): GREEN.** The op's only kernel, `device/kernels/dataflow/device_delay_spin.cpp`, performs **no data movement**: no `noc_async_*`, no CBs, no `InterleavedAddrGen`/`ShardedAddrGen`/addr-gen, no semaphores. It reads one compile-time arg (`get_compile_time_arg_val(0)`, line 11 — a sanctioned free function), `DPRINT`s, reads the wall-clock debug registers via `volatile` pointers, and calls the donor `spin()`. The donor `tt::data_movement::common::spin` (`data_movement/common/kernels/common.hpp:227-239`) likewise only reads the wall-clock registers. There are **no** Device-1.0 idioms and **no** CB-index free-function holdovers to migrate — the kernel is trivially Device-2.0 compliant. No table of violations.
- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | op has no CBs |
  | CBDescriptor `address_offset` (non-zero) | N/A | op has no CBs |
  | GlobalSemaphore | N/A | op has no semaphores |
  | Variable-count compile-time arguments (CTA varargs) | N/A | one CTA read at a constexpr index (`device_delay_spin.cpp:11`); fixed count |

  *Caveat:* Appendix A is authoritative for the enumerated features and none of them fires, so this subject is a clean all-`N/A`. The op's actual blocker (per-mesh-coord divergence + tensorless dispatch under `MetalV2FactoryConcept`) is **not** an Appendix A entry — it is a framework-capability gap surfaced at the concept transition and reported under Result. See Recipe notes: this is a candidate Appendix A / sheet-derivation addition.
- **CB endpoints:** N/A — the op declares no CBs. (Also skipped as a porter-only subject per the no-subset rule; see below.)
- **Offset base pointers: GREEN (N/A).** The op has no tensors and no address RTAs. The single kernel arg is the delay-cycle count (a plain `uint32_t` CTA), not a buffer address — no `->address()` fold anywhere.
- **TensorAccessor 3rd argument: GREEN (N/A).** No `TensorAccessor` is constructed anywhere (the op touches no tensor memory).

## Skipped subjects (whole-op RED, no portable subset)

Per the **Red** outcome scoping rule in `metal2_audit.md`, the seven purely-informational subjects are skipped on a whole-op RED with no portable subset; the op is re-audited against (possibly changed) code once the framework blocker clears. Recorded so the omission is not read as a clean result:

- **TTNN porting shape** — skipped: whole-op RED, no portable subset; re-audit on unblock. (Target would be `MetalV2FactoryConcept`, which is exactly what cannot carry this op.)
- **TensorParameter relaxations** — skipped: no portable subset; re-audit on unblock. (Sheet lists `relaxation = none` regardless.)
- **TensorParameter analysis** — skipped: no portable subset; re-audit on unblock. (Op has no tensors, so this would be trivially clean, but it is not owed now.)
- **CB endpoints** — skipped: no portable subset; re-audit on unblock. (Op has no CBs.)
- **RTA varargs** — skipped: no portable subset; re-audit on unblock.
- **Out-of-directory coupling** — skipped: no portable subset; re-audit on unblock. (The Device-2.0 gate already cleared the one donor, `common.hpp::spin`; no gate-relevant coupling is lost by deferring the full inventory.)
- **Incidental anomalies** — skipped as a dedicated pass; see Misc anomalies for the one item noticed in passing.

## Misc anomalies *(team-only, non-gating)*

- **`log_info(tt::LogAlways, ...)` on every dispatch.** `create_descriptor` logs at `LogAlways` on entry and exit (`apply_device_delay_device_operation.cpp:68,87`), and the host entry points log at `LogAlways` too (`apply_device_delay.cpp:29,31`; `device_operation.cpp:101`). For a *debug* utility this is arguably intentional, but `LogAlways` on the program-build path is noisy for any caller. Not porter work; noted for the ops team.

## Questions for the user *(none)*

## Recipe notes

- **Audit-method gap: the per-mesh-coordinate + tensorless framework wall is invisible to every gate the recipe enumerates.** This op cleanly passes the TTNN factory-concept gate (`Is able to port? = yes`, all cross-check columns match), the Device-2.0 gate, the full Appendix A feature scan, and both gating detail subjects (Offset base pointers, TensorAccessor 3rd arg) — yet it is genuinely un-portable, because its target concept `MetalV2FactoryConcept` (a) has no per-coordinate `create_program_artifacts` variant (`mesh_device_operation_adapter.hpp:887,907-913` stamps one spec on all coords) and (b) `TT_FATAL`s on a tensorless op (`:876-881`). The recipe's *TTNN porting shape* subject maps a `descriptor` op to `MetalV2FactoryConcept` unconditionally and does not check whether the op relies on per-coord divergence (via the `create_descriptor(..., mesh_dispatch_coordinate)` variant) or is tensorless. Suggest either: (i) a new Appendix A entry — *"per-mesh-coordinate program divergence (`create_descriptor` `mesh_dispatch_coordinate` variant) / tensorless op — UNSUPPORTED under `MetalV2FactoryConcept`"* with recognition signals `create_descriptor` taking `std::optional<ttnn::MeshCoordinate>` and `tensor_args_t` being empty; and/or (ii) a conjunct in the readiness sheet's `Is able to port?` derivation. This mirrors a previously observed "audit said GREEN, port found a framework wall" case for genuine per-coord ops (the `MetalV2FactoryConcept`/mesh-workload mutual exclusion at `operation_concepts.hpp:91-92`). Without such a rule, a lenient auditor could grade this op GREEN off the sheet and send a doomed port.
- **Readiness sheet conflict is not in the cross-check columns.** The recipe's "cross-check conflicts with the sheet → spreadsheet is broken" routing keys on the *cheaply-checkable factual columns*, all of which agree here. The disagreement is in the sheet's derived `Is able to port?` verdict vs. the actual framework capability — a class of sheet error the recipe's cross-check does not catch. Flagged to the sheet owner under Gate detail; noting here that the recipe's cross-check scope did not cover it.
