# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config`

- **`MoePaddingConfigDeviceOperation`** (`device/moe_padding_config_device_operation.hpp`)
  - `MeshWorkloadFactory` (`device/moe_padding_config_device_operation.cpp:329-359`): `create_mesh_workload` + `override_runtime_arguments`. It wraps the inner `ProgramFactory::create_descriptor` (`:248-327`) through `MeshDeviceOperationAdapter<…>::DescriptorMeshWorkloadAdapter` (`.hpp:66-82`). The op is in-place: it writes into the caller's `config` tensor.
- Kernel (owned by the op, the only one referenced): `device/kernels/dataflow/writer_moe_padding_config.cpp`. One data-movement kernel on one core.
- First audit of this op. No earlier `METAL2_PREPORT_AUDIT.md` exists.

**Scope:** TTNN op, Gen1 (WH/BH) target, within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Code audited at:** `main` `ad3ec49f6e2` (op dir identical on `edwinlee/topk_large_indices_descriptor_port`). Readiness sheet fetched 2026-11-06.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config` |
| **Overall** | **RED**: no portable subset |
| **DOps / Factories** | `MoePaddingConfigDeviceOperation` → `MeshWorkloadFactory` |
| *Prereqs*: Device 2.0 (every kernel used) | **Yes** |
| *Prereqs*: Cross-op escapes | skipped (whole-op RED, no portable subset; re-audit on unblock). The kernel includes only `tt_metal` API headers, so there is no donor to gate |
| *Feature Support*: overall | GREEN (no gate fired) |
| *Feature Support*: Variadic-CTA | Ok |
| *TTNN Readiness*: `Is able to port?` (the gate) | **No**: `Smuggled pointer (raw buffer addr in RTA/CRTA)` == `yes` (`Op Classification` == `Broken Op`) |
| *TTNN Readiness*: Concept (current) | `WorkloadDescriptor` per sheet (code: a `MeshWorkloadFactory` wrapping a `ProgramDescriptor` factory; see Questions) |
| *TTNN Readiness*: Secretly SPMD | Yes (sheet: "Per-coord RTAs"). Every coord gets the same program; only CRTA 0 `my_sp_coord` differs (`.cpp:264-265, 312-319`) |
| *TTNN Readiness*: Custom hash | Yes (not a gate; port leaves it intact): `compute_program_hash` @ `.hpp:91`, `.cpp:229-246` |
| *TTNN Readiness*: `get_dynamic_runtime_args` | No |
| *TTNN Readiness*: `override_runtime_arguments` | Yes: `MeshWorkloadFactory::override_runtime_arguments` @ `.cpp:338-359` (legacy `void(cached_mesh_workload_t&, …)` mesh-workload form; it exists *only* to patch the two smuggled CRTAs) |
| *TTNN Readiness*: Pybind `create_descriptor` | No |
| *TTNN Readiness*: Op-owned tensors | No |
| *TTNN Readiness*: Target concept | Not recorded (informational subject skipped; depends on how the TTNN fix lands) |
| *Port work*: Offset base pointer | none: GREEN |
| *Port work*: Tensor bindings | skipped (whole-op RED) |
| *TTNN Readiness*: TensorParameter relaxation | `none` (clears) |
| *Port work*: TensorAccessor 3rd arg | none: no accessor passes a 3rd argument |
| *Port work*: CB endpoints | skipped (whole-op RED) |

## Result

**RED → blocked on the TTNN factory concept gate only, routed to TTNN. No portable subset** (one factory, one kernel).

In plain terms, the kernel side is ready (Device 2.0 clean, no unsupported features, no offset or page-size problems). The block is host-side: the two 1-element metadata tensors `actual_start` / `actual_end` reach the kernel as **raw DRAM addresses in common runtime args** (`moe_padding_config_device_operation.cpp:316-318`), not as tensor bindings. The op keeps them correct on cache hits with a hand-written `override_runtime_arguments` that re-patches those two slots (`:338-359`). So it is *correct today*, but it is exactly the "smuggled pointer" shape the sheet blocks on.

**Path forward (TTNN, op-code side):** pass `actual_start` / `actual_end` through a binding the framework refreshes itself (as `config` already is: a `Buffer*` binding at `:323`). That lets the custom override and the `MeshWorkloadFactory` wrapper go. Once the sheet flips to `yes`, re-audit. Every other gate already clears, so the re-audit should be quick.

This is **not** the "stale #57409 PD-batch sheet row" case. #57409 did not touch this op (its last device-dir changes are #51624, #53997, #62567), and the sheet's `Smuggled pointer = yes` matches the code. So no waiver is offered.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED.** Row `experimental/deepseek_prefill/moe_padding_config` / `MoePaddingConfigDeviceOperation` / `MeshWorkloadFactory`: `Is able to port?` = `no`.
  - **Blocking column:** `Smuggled pointer (raw buffer addr in RTA/CRTA)` = `yes`; `Op Classification` = `Broken Op`. Route to **TTNN**. Evidence: `tensor_args.actual_start.buffer()->address()` and `tensor_args.actual_end.buffer()->address()` as CRTAs 3 and 4 (`.cpp:316-318`, both tagged `// smuggled-rta-ok`). The kernel reads them back raw (`writer_moe_padding_config.cpp:51-52`) and feeds them to `TensorAccessor(meta_args, addr)` (`:73, 83`). The refresh workaround is `override_runtime_arguments` writing `GetCommonRuntimeArgs(program, 0)[3..4]` on every cached program (`.cpp:349-357`).
  - **Primary-column cross-check:** `Custom hash` = `yes` ✓ · `get_dynamic_runtime_args` = `no` ✓ · `Override runtime args method?` = `yes` ✓ · `Pybind descriptor` = `no` ✓ (no `create_descriptor` / device-op `nb::class_` in `moe_padding_config_nanobind.cpp`) · `Secretly SPMD` = `yes` ✓ (structurally identical per-coord programs; only `my_sp_coord` varies) · `Op-owned tensors?` = `no` ✓ · factory set 1:1 ✓ · `Concept` = `WorkloadDescriptor`: consistent with the recipe's legend ("a mesh-workload return (`WorkloadDescriptor`)"), but see Questions. No invariant broken.

- **Device 2.0 (every kernel used):** **GREEN.** `writer_moe_padding_config.cpp` uses `Noc` (`:63`), `CircularBuffer` objects with method calls (`:70-71, 81, 87-88, 124-142`), `noc.async_read/async_write` with `TensorAccessor` (`:74, 84, 140`), `noc.async_*_barrier`, and `CoreLocalMem`. The one CB-index free function, `get_local_cb_interface(cb_id_out).fifo_page_size` (`:123`), is on the sanctioned list. No donor kernels; includes are `tt_metal` API headers only.

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | |
  | CBDescriptor `address_offset` (non-zero) | N/A | neither `CBDescriptor` sets it (`.cpp:277-295`) |
  | GlobalSemaphore | N/A | no semaphores of any kind |

- **CB endpoints:** skipped (whole-op RED, no portable subset; re-audit on unblock).

- **Offset base pointers:** **GREEN.** All three tensor addresses are bare bases: `config` as a `Buffer*` binding (`.cpp:323`), and `actual_start` / `actual_end` as bare `->address()` CRTAs with no folded offset (`:316-318`). The kernel reads page 0 at `offset_bytes = 0`. Not in `2026-07-19_offset_base_pointers.md`.

- **TensorAccessor 3rd argument:** **N/A.** No accessor passes a 3rd argument (`writer_moe_padding_config.cpp:73, 83, 139`).

## Port-work summary  *(mirrors the brief)*

No brief is issued (RED). Tensor bindings, CB endpoints, TensorAccessor 3rd arg (none), relaxation (`none`): the binding inventory is deferred to the re-audit, because the TTNN fix changes exactly how `actual_start` / `actual_end` are delivered.

## Heads-ups  *(mirrors the brief)*

Skipped (whole-op RED, no portable subset; re-audit on unblock). For the re-audit:

- **One accessor serves two tensors.** A single `TensorAccessorArgs` built from `actual_start` (`.cpp:301`) is used for both `actual_start` and `actual_end` reads (`writer_moe_padding_config.cpp:58, 73, 83`), pinned by `validate_runtime_args` (`.cpp:192-197`). In Metal 2.0 each tensor gets its own binding, so this sharing goes away.
- **Per-coord CRTA.** `my_sp_coord` is the only per-device difference (`.cpp:264-265`). This decides whether the target is single-program with per-coord run args, or `MeshWorkloadSpecFactoryConcept` (see the dispatch audit's Recipe notes).

## Team-only

- **Out-of-directory coupling & donor shape:** skipped (whole-op RED, no portable subset). No out-of-dir kernel includes were seen.
- **Relaxation candidates** (FALLIBLE): the custom hash keys `config`'s dtype/layout/memory_config/padded_shape and `actual_start`'s memory_config, but not the metadata tensors' shape or dtype (`.cpp:235-245`). Those are pinned to `UINT32`, 1 element, by validation instead. Nothing to relax beyond strict.
- **TTNN factory analysis:** see the Status summary. Custom hash present (leave as is). `override_runtime_arguments` present, but its only job is the smuggled-CRTA refresh, so it likely disappears with the TTNN fix rather than being translated.
- **Why the informational subjects were skipped:** the only blocker is cleared on the **op-code side** (TTNN changes how the metadata tensors are bound), which rewrites the bindings and runtime args those subjects describe.

## Questions for the user

1. **`Concept` label vs. code shape:** the sheet says `WorkloadDescriptor`, but the factory has no `create_workload_descriptor`. It is a `MeshWorkloadFactory` (`create_mesh_workload` + `cached_mesh_workload_t`, so it satisfies `MeshWorkloadFactoryConcept`, not `ProgramDescriptorFactoryConcept`) that wraps a `ProgramDescriptor` factory via `DescriptorMeshWorkloadAdapter`. The recipe's legend accepts "a mesh-workload return" as `WorkloadDescriptor`, so I did not treat this as a broken sheet. Worth confirming with the sheet owner that this hybrid is meant to be labelled `WorkloadDescriptor`.

## Recipe notes

- The `Concept` cross-check legend ("a mesh-workload return (`WorkloadDescriptor`)") doesn't separate `create_workload_descriptor` (a true `ProgramDescriptorFactoryConcept`) from a `MeshWorkloadFactoryConcept` that wraps a descriptor through `DescriptorMeshWorkloadAdapter`. The two satisfy different concepts in `operation_concepts.hpp`. A sentence naming this hybrid would remove the judgment call.
